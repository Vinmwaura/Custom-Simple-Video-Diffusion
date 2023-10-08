import os
import csv
import json
import pathlib
import logging
import argparse

import cv2

import torch
import torchvision
import torch.nn.functional as F

# U Net Model.
from models.Video_U_Net import Video_U_Net

from diffusion_enums import *

# Degradation Operators.
from degraders import *

from utils.utils import *

from diffusion_sampling import (
    ddpm_sampling,
    ddim_sampling,
    cold_diffusion_sampling)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    project_name = "Video-Diffusion"

    parser = argparse.ArgumentParser(
        description="Train Video Diffusion models.")
    
    parser.add_argument(
        "-c",
        "--config-path",
        help="File path to load json config file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--device",
        help="Which hardware device will model run on (default='cpu')?",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    
    args = vars(parser.parse_args())

    # Device to run model on.
    device = args["device"]

    # Load and Parse config JSON.
    config_json = args["config_path"]
    with open(config_json, 'r') as json_file:
        json_data = json_file.read()
    config_dict = json.loads(json_data)

    # Global steps in between checkpoints
    checkpoint_steps = config_dict["checkpoint_steps"]
    
    # Global steps in between halving learning rate.
    lr_steps = config_dict["lr_steps"]
    max_epoch = config_dict["max_epoch"]
    plot_img_count = config_dict["plot_img_count"]
    
    # Regex to list of images or json containing labelled dataset.
    csv_path = config_dict["csv_path"]
    if csv_path is None:
        raise ValueError("No csv path entered.")
    
    # Output Directory for model's checkpoint, logs and sample output.
    out_dir = config_dict["out_dir"]
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    # File path to checkpoints.
    diffusion_checkpoint = config_dict["model_checkpoint"]
    config_checkpoint = config_dict["config_checkpoint"]

    # Model Params.
    diffusion_lr = config_dict["diffusion_lr"]
    batch_size = config_dict["batch_size"]

    # Diffusion Params.
    # Noise Schedulers (LINEAR, COSINE).
    if config_dict["noise_scheduler"] == "LINEAR":
        noise_scheduling = NoiseScheduler.LINEAR

        # Noise Scheduler Params.
        beta_1 = config_dict["beta1"]
        beta_T = config_dict["betaT"]
    elif config_dict["noise_scheduler"] == "COSINE":
        noise_scheduling = NoiseScheduler.COSINE
    else:
        raise ValueError("Invalid noise scheduler type.")

    if config_dict["diffusion_alg"] == "DDIM":
        diffusion_alg = DiffusionAlg.DDIM
    elif config_dict["diffusion_alg"] == "DDPM":
        diffusion_alg = DiffusionAlg.DDPM
    elif config_dict["diffusion_alg"] == "COLD":
        diffusion_alg = DiffusionAlg.COLD
    else:
        raise ValueError("Invalid diffusion algorithm type.")

    min_noise_step = config_dict["min_noise_step"]  # t_1
    max_noise_step = config_dict["max_noise_step"]  # T
    max_actual_noise_step = config_dict["max_actual_noise_step"]  # Max timestep used in training step (For ensemble models training).
    skip_step = config_dict["skip_step"]  # Step to be skipped when sampling.
    if max_actual_noise_step < min_noise_step \
        or max_noise_step < min_noise_step \
            or skip_step > max_actual_noise_step \
                or skip_step < 0 \
                    or min_noise_step < 0:
        raise ValueError("Invalid step values entered!")

    log_path = os.path.join(out_dir, f"{project_name}.log")
    logging.basicConfig(
        # filename=log_path,
        format="%(asctime)s %(message)s",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        level=logging.DEBUG)

    # Initialize gradient scaler.
    scaler = torch.cuda.amp.GradScaler()
    
    # Video Params.
    frame_window = config_dict["frame_window"]
    frame_skipped = config_dict["frame_skipped"]
    low_res_dim = config_dict["low_res_dim"]

    # Low Resolution Dimension, used for Super Resolution Models.
    if low_res_dim is not None and not isinstance(low_res_dim, int):
        raise ValueError("Invalid Image Dimension, must be int.")

    if low_res_dim is not None and low_res_dim <= 2:
        raise ValueError("Invalid Image Dimension, must be greater than 2.")

    # Dataset and DataLoader.
    # Custom Image Dataset Loader.
    from dataset_loader.video_frames_dataset import VideoFramesDataset
    dataset = VideoFramesDataset(
        csv_path,
        frame_window=frame_window,
        frames_skipped=frame_skipped)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)

    # Model Params.
    in_channel = config_dict["in_channel"]
    out_channel = config_dict["out_channel"]
    mapping_channel = config_dict["mapping_channel"]
    num_layers = config_dict["num_layers"]
    num_resnet_block = config_dict["num_resnet_block"]
    attn_layers = config_dict["attn_layers"]
    attn_heads = config_dict["attn_heads"]
    attn_dim_per_head = config_dict["attn_dim_per_head"]
    time_dim = config_dict["time_dim"]
    cond_dim = config_dict["cond_dim"]
    min_channel = config_dict["min_channel"]
    max_channel = config_dict["max_channel"]
    img_recon = config_dict["img_recon"]

    cond_dim_combined = None if cond_dim is None else \
        cond_dim * (frame_window // (frame_skipped + 1))

    # Model.
    diffusion_net = Video_U_Net(
        in_channel=in_channel,
        out_channel=out_channel,
        mapping_channel=mapping_channel,
        num_layers=num_layers,
        num_resnet_blocks=num_resnet_block,
        attn_layers=attn_layers,
        num_heads=attn_heads,
        dim_per_head=attn_dim_per_head,
        time_dim=time_dim,
        cond_dim=cond_dim_combined,
        min_channel=min_channel,
        max_channel=max_channel,
        image_recon=img_recon)
    
    # Training has Labels or Super Resolution.
    has_labels = config_dict["has_labels"]
    super_resolution_training = config_dict["super_resolution_training"]

    plot_frames = None
    plot_video_labels = None
    if has_labels or super_resolution_training:
        plot_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=plot_img_count,
            num_workers=1,
            shuffle=True)

        plot_frames, plot_video_labels = next(iter(plot_dataloader))

    # Conditional Information i.e labels.
    if has_labels:
        if cond_dim is None:
            raise ValueError("Invalid Cond Dim parameter!")
        
        if plot_video_labels.shape[1] == 0:
            raise ValueError("No Labels passed!")

        labels_path = os.path.join(out_dir, "labels.txt")

        with open(csv_path, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)

        all_rows = [header]

        plot_video_labels_list = plot_video_labels.cpu().tolist()
        for video_idx, plot_video_label in enumerate(plot_video_labels_list):
            all_rows.append([f"Video-{video_idx+1:,}"])

            # Group labels per frames in each video.
            plot_frames_label = list(chunks(plot_video_label, cond_dim))
            for frame_idx, plot_frame_label in enumerate(plot_frames_label):
                labels = [f"Frame-{frame_idx + 1:,}"] + plot_frame_label
                all_rows.append(labels)

        with open(labels_path, "a") as f:
            wr = csv.writer(f)
            wr.writerows(all_rows)

    # For Super-Resolution Training, generate GIF of low-res frames and high-res frames.
    plot_low_res_frames = None
    if super_resolution_training:
        if low_res_dim is None:
            raise ValueError("Invalid Low Resolution Dim parameter!")

        _, _, plot_F, _, _ = plot_frames.shape
        plot_low_res_frames = F.interpolate(
            plot_frames,
            size=(plot_F, low_res_dim, low_res_dim))

        make_gif(
            plot_frames,
            global_steps=-1,
            dest_path=out_dir,
            log=print)

        make_gif(
            plot_low_res_frames,
            global_steps=-2,
            dest_path=out_dir,
            log=print)

    # Load Pre-trained optimization configs, ignored if no checkpoint is passed.
    load_diffusion_optim = config_dict["load_diffusion_optim"]

    # Load Diffusion Model Checkpoints.
    if diffusion_checkpoint is not None:
        diffusion_status, diffusion_dict= load_checkpoint(diffusion_checkpoint)
        if not diffusion_status:
            raise Exception("An error occured while loading model checkpoint!")
        diffusion_net.custom_load_state_dict(diffusion_dict["model"])
        diffusion_net = diffusion_net.to(device)
        
        diffusion_optim = torch.optim.Adam(
            diffusion_net.parameters(),
            lr=diffusion_lr,
            betas=(0.5, 0.999))

        if load_diffusion_optim:
            diffusion_optim.load_state_dict(diffusion_dict["optimizer"])
    else:
        diffusion_net = diffusion_net.to(device)
        
        diffusion_optim = torch.optim.Adam(
            diffusion_net.parameters(),
            lr=diffusion_lr,
            betas=(0.5, 0.999))

    # Load Config Checkpoints.
    if config_checkpoint is not None:
        config_ckpt_status, config_ckpt_dict = load_checkpoint(config_checkpoint)
        if not config_ckpt_status:
            raise Exception("An error occured while loading config checkpoint!")

        if noise_scheduling == NoiseScheduler.LINEAR:
            beta_1 = config_ckpt_dict["beta_1"]
            beta_T = config_ckpt_dict["beta_T"]

        # Training Params.
        starting_epoch = config_ckpt_dict["starting_epoch"]
        global_steps = config_ckpt_dict["global_steps"]
    else:
        # Training Params.
        starting_epoch = 0
        global_steps = 0

    # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
    if noise_scheduling == NoiseScheduler.LINEAR:
        noise_degradation = NoiseDegradation(
            beta_1,
            beta_T,
            max_noise_step,
            device)
    elif noise_scheduling == NoiseScheduler.COSINE:
        noise_degradation = CosineNoiseDegradation(max_noise_step)

    logging.info("#" * 100)
    logging.info(f"Video Parameters:")
    logging.info(f"Video Frame Window: {frame_window:,}")
    logging.info(f"Video Frame Skipped: {frame_skipped:,}")
    logging.info("#" * 100)
    logging.info(f"Train Parameters:")
    logging.info(f"Has Labels: {has_labels}")
    logging.info(f"Training Super Resolution: {super_resolution_training}")
    logging.info(f"Max Epoch: {max_epoch:,}")
    logging.info(f"Dataset CSV Path: {csv_path}")
    logging.info(f"Output Path: {out_dir}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps}")
    logging.info(f"Batch size: {batch_size:,}")
    logging.info(f"Diffusion LR: {diffusion_optim.param_groups[0]['lr']:.5f}")
    logging.info(f"Low Resolution Dim (For Super Resolution): {low_res_dim}")
    logging.info("#" * 100)
    logging.info(f"Model Parameters:")
    logging.info(f"In Channel: {in_channel:,}")
    logging.info(f"Out Channel: {out_channel:,}")
    logging.info(f"Mapping Channel: {mapping_channel}")
    logging.info(f"Num Layers: {num_layers:,}")
    logging.info(f"Num Resnet Block: {num_resnet_block:,}")
    logging.info(f"Attn Layers: {attn_layers}")
    logging.info(f"Attn Heads: {attn_heads:,}")
    logging.info(f"Attn dim per head: {attn_dim_per_head}")
    logging.info(f"Time Dim: {time_dim:,}")
    logging.info(f"Cond Dim: {cond_dim}")
    logging.info(f"Min Channel: {min_channel:,}")
    logging.info(f"Max Channel: {max_channel:,}")
    logging.info(f"Img Recon: {img_recon}")
    logging.info("#" * 100)
    logging.info(f"Diffusion Parameters:")
    if noise_scheduling == NoiseScheduler.LINEAR:
        logging.info(f"Beta_1: {beta_1:,.5f}")
        logging.info(f"Beta_T: {beta_T:,.5f}")
    logging.info(f"Min Noise Step: {min_noise_step:,}")
    logging.info(f"Max Noise Step: {max_noise_step:,}")
    logging.info(f"Max Actual Noise Step: {max_actual_noise_step:,}")
    logging.info("#" * 100)

    for epoch in range(starting_epoch, max_epoch):
        # Diffusion Loss.
        total_diffusion_loss = 0

        # Number of iterations.
        training_count = 0

        for index, (video_frames, frame_labels) in enumerate(dataloader):
            training_count += 1

            video_frames = video_frames.to(device)
            frame_labels = frame_labels.to(device)

            video_N, video_C, video_F, video_H, video_W = video_frames.shape

            # For Super Resolution Training.
            low_res_video_frames = None
            if super_resolution_training:
                _, _, low_res_F, _, _ = video_frames.shape
                low_res_video_frames = F.interpolate(
                    video_frames,
                    size=(low_res_F, low_res_dim, low_res_dim))

                low_res_video_frames = low_res_video_frames.to(device)

            # eps Noise.
            noise = torch.randn_like(video_frames)

            #################################################
            #             Diffusion Training.               #
            #################################################
            diffusion_optim.zero_grad()

            # Random Noise Step(t).
            rand_noise_step = torch.randint(
                low=min_noise_step,
                high=max_actual_noise_step,
                size=(video_N, ),
                device=device)

            # Train model.
            diffusion_net.train()

            # TODO: Allow for toggling in cases of Hardware that don't support this.
            # Enable autocasting for mixed precision.
            with torch.cuda.amp.autocast():
                # Noise degraded image (x_t).
                # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                x_t_frames = noise_degradation(
                    img=video_frames,
                    steps=rand_noise_step,
                    eps=noise)

                if diffusion_alg == DiffusionAlg.COLD:
                    # Predicts image reconstruction.
                    x0_approx = diffusion_net(
                        x=x_t_frames,
                        t=rand_noise_step,
                        cond=None if frame_labels.shape[1] == 0 else frame_labels,
                        lr_img=low_res_video_frames)

                    diffusion_loss = F.mse_loss(
                        x0_approx,
                        video_frames)
                else:
                    # Predicts noise from x_t.
                    # eps_param(sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps, t). 
                    noise_approx = diffusion_net(
                        x=x_t_frames,
                        t=rand_noise_step,
                        cond=None if frame_labels.shape[1] == 0 else frame_labels,
                        lr_img=low_res_video_frames)

                    # Simplified Training Objective.
                    # L_simple(param) = E[||eps - eps_param(sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps, t).||^2]
                    diffusion_loss = F.mse_loss(
                        noise_approx,
                        noise)
                
                if torch.isnan(diffusion_loss):
                    raise Exception("NaN encountered during training")

            # Scale the loss and do backprop.
            scaler.scale(diffusion_loss).backward()
            
            # Update the scaled parameters.
            scaler.step(diffusion_optim)

            # Update the scaler for next iteration
            scaler.update()
            
            total_diffusion_loss += diffusion_loss.item()

            if global_steps % lr_steps == 0 and global_steps > 0:
                # Update Diffusion LR.
                for diffusion in diffusion_optim.param_groups:
                    diffusion['lr'] = diffusion['lr'] * 0.5

            # Checkpoint and Plot Images.
            if global_steps % checkpoint_steps == 0 and global_steps >= 0:
                config_state = {
                    "starting_epoch": starting_epoch,
                    "global_steps": global_steps}
                
                if noise_scheduling == NoiseScheduler.LINEAR:
                    config_state["beta_1"] = beta_1
                    config_state["beta_T"] = beta_T
                
                # Save Config Net.
                save_model(
                    model_net=config_state,
                    file_name="config",
                    dest_path=out_dir,
                    checkpoint=True,
                    steps=global_steps)

                # Save Diffusion Net.
                diffusion_state = {
                    "model": diffusion_net.state_dict(),
                    "optimizer": diffusion_optim.state_dict(),}
                save_model(
                    model_net=diffusion_state,
                    file_name="diffusion",
                    dest_path=out_dir,
                    checkpoint=True,
                    steps=global_steps)
                
                # X_T ~ N(0, I).
                noise = torch.randn((
                    plot_img_count,
                    video_C,
                    video_F,
                    video_H,
                    video_W), device=device)
                x_t_frames_plot = 1 * noise

                if super_resolution_training:
                    plot_low_res_frames = plot_low_res_frames.to(device)

                if has_labels:
                    plot_video_labels = plot_video_labels.to(device)

                if diffusion_alg == DiffusionAlg.DDPM:
                    x0_approx = ddpm_sampling(
                        diffusion_net,
                        noise_degradation,
                        x_t_frames_plot,
                        min_noise_step,
                        max_actual_noise_step,
                        lr_img=plot_low_res_frames,
                        labels_tensor=plot_video_labels,
                        device=device,
                        log=print)
                elif diffusion_alg == DiffusionAlg.DDIM:
                    x0_approx = ddim_sampling(
                        diffusion_net,
                        noise_degradation,
                        x_t_frames_plot,
                        min_noise=min_noise_step,
                        max_noise=max_actual_noise_step,
                        lr_img=plot_low_res_frames,
                        labels_tensor=plot_video_labels,
                        ddim_step_size=skip_step,
                        device=device,
                        log=print)
                elif diffusion_alg == DiffusionAlg.COLD:
                    x0_approx = cold_diffusion_sampling(
                        diffusion_net,
                        noise_degradation,
                        x_t_frames_plot,
                        noise,
                        min_noise=min_noise_step,
                        max_noise=max_actual_noise_step,
                        lr_img=plot_low_res_frames,
                        labels_tensor=plot_video_labels,
                        skip_step_size=10,
                        device=device,
                        log=print)
                else:
                    raise ValueError("Invalid Diffusion Algorithm!")

                make_gif(
                    x0_approx,
                    global_steps=global_steps,
                    dest_path=out_dir,
                    log=print)

            temp_avg_diffusion = total_diffusion_loss / training_count
            message = "Cum. Steps: {:,} | Steps: {:,} / {:,} | Diffusion: {:.5f} | LR: {:.9f}".format(
                global_steps + 1,
                index + 1,
                len(dataloader),
                temp_avg_diffusion, 
                diffusion_optim.param_groups[0]['lr']
            )
            logging.info(message)

            global_steps += 1

        # Save Config Net.
        config_state = {
            "starting_epoch": starting_epoch,
            "global_steps": global_steps
        }
        if noise_scheduling == NoiseScheduler.LINEAR:
            config_state["beta_1"] = beta_1
            config_state["beta_T"] = beta_T
        save_model(
            model_net=config_state,
            file_name="config",
            dest_path=out_dir,
            checkpoint=True,
            steps=global_steps)
        
        # Save Diffusion Net.
        diffusion_state = {
            "model": diffusion_net.state_dict(),
            "optimizer": diffusion_optim.state_dict(),}
        save_model(
            model_net=diffusion_state,
            file_name="diffusion",
            dest_path=out_dir,
            checkpoint=True,
            steps=global_steps)
        
        avg_diffusion = total_diffusion_loss / training_count
        message = "Epoch: {:,} | Diffusion: {:.5f} | LR: {:.9f}".format(
            epoch,
            avg_diffusion,
            diffusion_optim.param_groups[0]['lr']
        )
        logging.info(message)

if __name__ == "__main__":
    main()

import os
import csv
import json
import pathlib
import argparse

import torch
import torchvision
import torch.nn.functional as F

# Degradation Operators.
from degraders import *

from utils.utils import *

from diffusion_enums import *

from diffusion_sampling import (
    ddpm_sampling,
    ddim_sampling,
    cold_diffusion_sampling)

# U Net Model.
from models.Video_U_Net import Video_U_Net

def get_noise_degradation_algorithm(noise_scheduling, model_dict, device):
    # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
    if noise_scheduling == NoiseScheduler.LINEAR:
        noise_degradation = NoiseDegradation(
            model_dict["beta1"],
            model_dict["betaT"],
            model_dict["max_noise_step"],
            device)
    elif noise_scheduling == NoiseScheduler.COSINE:
        noise_degradation = CosineNoiseDegradation(
            model_dict["max_noise_step"])
    return noise_degradation

def generate_video(
        device,
        noise,
        diffusion_net,
        diffusion_alg,
        noise_degradation,
        model_dict,
        low_res_image,
        plot_video_labels,
        skip_step):
    # Initial x_t ~ Noise.
    x_t_frames_plot = 1 * noise

    if diffusion_alg == DiffusionAlg.DDPM:
        x0_approx = ddpm_sampling(
            diffusion_net,
            noise_degradation,
            x_t_frames_plot,
            model_dict["min_noise_step"],
            model_dict["max_actual_noise_step"],
            cond_image=low_res_image,
            cond_labels=plot_video_labels,
            device=device,
            log=print)
    elif diffusion_alg == DiffusionAlg.DDIM:
        x0_approx = ddim_sampling(
            diffusion_net,
            noise_degradation,
            x_t_frames_plot,
            min_noise=model_dict["min_noise_step"],
            max_noise=model_dict["max_actual_noise_step"],
            cond_image=low_res_image,
            cond_labels=plot_video_labels,
            ddim_step_size=skip_step,
            device=device,
            log=print)
    elif diffusion_alg == DiffusionAlg.COLD:
        x0_approx = cold_diffusion_sampling(
            diffusion_net,
            noise_degradation,
            x_t_frames_plot,
            noise,
            min_noise=model_dict["min_noise_step"],
            max_noise=model_dict["max_actual_noise_step"],
            cond_image=low_res_image,
            cond_labels=plot_video_labels,
            skip_step_size=10,
            device=device,
            log=print)
    else:
        raise ValueError("Invalid Diffusion Algorithm!")

    return x0_approx

def create_model(model_dict):
    # Model Params.
    in_channel = model_dict["in_channel"]
    out_channel = model_dict["out_channel"]
    mapping_channel = model_dict["mapping_channel"]
    num_layers = model_dict["num_layers"]
    num_resnet_block = model_dict["num_resnet_block"]
    attn_layers = model_dict["attn_layers"]
    attn_heads = model_dict["attn_heads"]
    attn_dim_per_head = model_dict["attn_dim_per_head"]
    time_dim = model_dict["time_dim"]
    cond_dim = model_dict["cond_dim"]
    min_channel = model_dict["min_channel"]
    max_channel = model_dict["max_channel"]
    img_recon = model_dict["img_recon"]

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
        cond_dim=cond_dim,
        min_channel=min_channel,
        max_channel=max_channel,
        image_recon=img_recon)
    
    return diffusion_net

def main():
    parser = argparse.ArgumentParser(
        description="Generate Videos.")
    
    parser.add_argument(
        "-c",
        "--config",
        help="File path to Model(s) config file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "-l",
        "--labels",
        nargs="*",
        help="Conditional Labels.",
        type=float,
        default=None)
    parser.add_argument(
        "--out-dir",
        help="File path to save output.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--seed",
        help="Seed value.",
        default=None,
        type=int)
    parser.add_argument(
        "--device",
        help="Which hardware device will model run on (default='cpu')?",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    
    args = vars(parser.parse_args())

    # Seed Value.
    seed_val = args["seed"]
    if seed_val is not None:
        torch.manual_seed(seed_val)

    # Device to run model on.
    device = args["device"]

    # Output Path.
    out_dir = args["out_dir"]

    # Loads model details from json file.
    with open(args["config"], "r") as f:
        models_details = json.load(f)
    
    noise_scheduling_dict = {
        "LINEAR": NoiseScheduler.LINEAR,
        "COSINE": NoiseScheduler.COSINE}
    diffusion_alg_dict = {
        "DDIM": DiffusionAlg.DDIM,
        "DDPM": DiffusionAlg.DDPM,
        "COLD": DiffusionAlg.COLD}

    low_res_image = None

    for model_dict in models_details["models"]:
        # Diffusion Params
        min_noise_step = model_dict["min_noise_step"]  # t_1
        max_noise_step = model_dict["max_noise_step"]  # T
        max_actual_noise_step = model_dict["max_actual_noise_step"]  # Max timestep used in training step (For ensemble models training).
        skip_step = model_dict["skip_step"]  # Step to be skipped when sampling.
        if max_actual_noise_step < min_noise_step \
            or max_noise_step < min_noise_step \
                or skip_step > max_actual_noise_step \
                    or skip_step < 0 \
                        or min_noise_step < 0:
            raise ValueError("Invalid step values entered!")

        if model_dict["frame_skipped"] > 0:
            total_frames = 0
            for frame_count in range(model_dict["frame_window"]):
                if frame_count % model_dict["frame_skipped"] == 0:
                    total_frames += 1
        else:
            total_frames = model_dict["frame_window"]
        
        # Get Noise Scheduler.
        noise_scheduling = noise_scheduling_dict[
            model_dict["noise_scheduler"]]
        
        # Get Diffusion Algorithm.
        diffusion_alg = diffusion_alg_dict[
            model_dict["diffusion_alg"]]
        
        # Noise Degradation Algorithms
        noise_degradation = get_noise_degradation_algorithm(
            noise_scheduling,
            model_dict,
            device)

        # Create Diffusion Model.
        diffusion_net = create_model(model_dict)

        # Load model checkpoints.
        diffusion_status, diffusion_dict= load_checkpoint(model_dict["model_checkpoint"])
        if not diffusion_status:
            raise Exception("An error occured while loading model checkpoint!")
        diffusion_net.custom_load_state_dict(diffusion_dict["model"])
        diffusion_net = diffusion_net.to(device)

        # X_T ~ N(0, I).
        noise = torch.randn((
            1,
            model_dict["img_channel"],
            total_frames,
            model_dict["img_dim"],
            model_dict["img_dim"]),
        device=device)

        # Label Dimensions.
        if model_dict["cond_dim"] is not None:
            if args["labels"] is None:
                raise ValueError("Invalid / No conditional labels passed!")
            plot_video_labels = torch.tensor(args["labels"]).float().to(args["device"])
        else:
            plot_video_labels = None
        
        # Base Model.
        if low_res_image is None:
            plot_video_labels = plot_video_labels.reshape(
                (1, total_frames, model_dict["cond_dim"]))
            
            x0_approx = generate_video(
                device,
                noise,
                diffusion_net,
                diffusion_alg,
                noise_degradation,
                model_dict,
                low_res_image,
                plot_video_labels,
                skip_step)
            
            make_gif(
                x0_approx,
                global_steps=model_dict["img_dim"],
                dest_path=out_dir,
                log=print)

            low_res_image = x0_approx

            # Clip values to be between -1 and 1 to avoid artefacts.
            low_res_image = np.clip(low_res_image, -1, 1)
        else:
            # Upsampling Models.
            _, _, F, _, _ = low_res_image.shape
            batch_count = F // total_frames
            combined_img = None
            for i in range(0, batch_count):
                cond_dim = model_dict["cond_dim"]

                cond_img = low_res_image[
                    :,
                    :,
                    i * total_frames:(i + 1) * total_frames,
                    :,
                    :]
                cond_labels = plot_video_labels[i * total_frames * cond_dim:(i + 1) * total_frames * cond_dim]
                cond_labels = cond_labels.reshape((1, total_frames, cond_dim))
                
                x0_approx = generate_video(
                    device,
                    noise,
                    diffusion_net,
                    diffusion_alg,
                    noise_degradation,
                    model_dict,
                    cond_img,
                    cond_labels,
                    skip_step)

                if combined_img is None:
                    combined_img = x0_approx
                else:
                    combined_img = torch.cat((combined_img, x0_approx), dim=2)

            make_gif(
                combined_img,
                global_steps=model_dict["img_dim"],
                dest_path=out_dir,
                log=print)

            # Clip values to be between -1 and 1 to avoid artefacts.
            low_res_image = np.clip(low_res_image, -1, 1)

if __name__ == "__main__":
    main()

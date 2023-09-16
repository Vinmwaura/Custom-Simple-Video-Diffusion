import os
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
    frame_interpolation_sampling)

# U Net Model.
from models.Video_U_Net import Video_U_Net

def generate_base_frames(
        device,
        config_dict):
    # Image Params.
    img_N = config_dict["img_N"]
    img_C = config_dict["img_C"]
    img_F = config_dict["img_F"]
    img_H = config_dict["img_H"]
    img_W = config_dict["img_W"]

    # Sampling Params.
    base_beta_1 = config_dict["base_beta_1"]
    base_beta_T = config_dict["base_beta_T"]
    base_skip_step = config_dict["base_skip_step"]
    base_max_noise_step = config_dict["base_max_noise_step"]
    base_min_noise_step = config_dict["base_min_noise_step"]

    if config_dict["base_diffusion_alg"] == "DDIM":
        diffusion_alg = DiffusionAlg.DDIM
    elif config_dict["base_diffusion_alg"] == "DDPM":
        diffusion_alg = DiffusionAlg.DDPM
    else:
        raise ValueError("Invalid diffusion algorithm type.")

    # Degradation Algorithm.
    base_noise_degradation = NoiseDegradation(
        base_beta_1,
        base_beta_T,
        base_max_noise_step,
        device)
    
    # Base Model Params.
    base_model_checkpoint = config_dict["base_model_checkpoint"]

    base_in_channel = config_dict["base_in_channel"]
    base_out_channel = config_dict["base_out_channel"]
    base_num_layers = config_dict["base_num_layers"]
    base_num_resnet_block = config_dict["base_num_resnet_block"]
    base_attn_layers = config_dict["base_attn_layers"]
    base_attn_heads = config_dict["base_attn_heads"]
    base_attn_dim_per_head = config_dict["base_attn_dim_per_head"]
    base_time_dim = config_dict["base_time_dim"]
    base_cond_dim = config_dict["base_cond_dim"]
    base_min_channel = config_dict["base_min_channel"]
    base_max_channel = config_dict["base_max_channel"]
    base_img_recon = config_dict["base_img_recon"]
    
    # Base Model.
    base_diffusion_net = Video_U_Net(
        in_channel=base_in_channel,
        out_channel=base_out_channel,
        num_layers=base_num_layers,
        num_resnet_blocks=base_num_resnet_block,
        attn_layers=base_attn_layers,
        num_heads=base_attn_heads,
        dim_per_head=base_attn_dim_per_head,
        time_dim=base_time_dim,
        cond_dim=base_cond_dim,
        min_channel=base_min_channel,
        max_channel=base_max_channel,
        image_recon=base_img_recon)
    
    base_diffusion_status, base_diffusion_dict= load_checkpoint(base_model_checkpoint)
    if not base_diffusion_status:
        raise Exception("An error occured while loading base model checkpoint!")
    base_diffusion_net.custom_load_state_dict(base_diffusion_dict["model"])
    base_diffusion_net = base_diffusion_net.to(device)
    
    # Generate Base Frames.
    noise = torch.randn((img_N, img_C, img_F, img_H, img_W), device=device)
    x_t_frames = 1 * noise

    if diffusion_alg == DiffusionAlg.DDPM:
        x0_frames_approx = ddpm_sampling(
            base_diffusion_net,
            base_noise_degradation,
            x_t_frames,
            base_min_noise_step,
            base_max_noise_step,
            cond_img=None,
            labels_tensor=None,
            device=device,
            log=print)
    elif diffusion_alg == DiffusionAlg.DDIM:
        x0_frames_approx = ddim_sampling(
            base_diffusion_net,
            base_noise_degradation,
            x_t_frames,
            min_noise=base_min_noise_step,
            max_noise=base_max_noise_step,
            cond_img=None,
            labels_tensor=None,
            ddim_step_size=base_skip_step,
            device=device,
            log=print)
    
    return x0_frames_approx

def generate_interpolated_frames(
        device,
        base_video_frames,
        config_dict):

    # Sampling Params.
    interpolation_skip_step = config_dict["interpolation_skip_step"]
    interpolation_max_noise_step = config_dict["interpolation_max_noise_step"]
    interpolation_min_noise_step = config_dict["interpolation_min_noise_step"]

    # Degradation Algorithm.
    interpolation_noise_degradation = CosineNoiseDegradation(interpolation_max_noise_step)

    # interpolation Model Params.
    interpolation_model_checkpoint = config_dict["interpolation_model_checkpoint"]

    interpolation_in_channel = config_dict["interpolation_in_channel"]
    interpolation_out_channel = config_dict["interpolation_out_channel"]
    interpolation_num_layers = config_dict["interpolation_num_layers"]
    interpolation_num_resnet_block = config_dict["interpolation_num_resnet_block"]
    interpolation_attn_layers = config_dict["interpolation_attn_layers"]
    interpolation_attn_heads = config_dict["interpolation_attn_heads"]
    interpolation_attn_dim_per_head = config_dict["interpolation_attn_dim_per_head"]
    interpolation_time_dim = config_dict["interpolation_time_dim"]
    interpolation_cond_dim = config_dict["interpolation_cond_dim"]
    interpolation_min_channel = config_dict["interpolation_min_channel"]
    interpolation_max_channel = config_dict["interpolation_max_channel"]
    interpolation_img_recon = config_dict["interpolation_img_recon"]
    
    # interpolation Model.
    interpolation_diffusion_net = Video_U_Net(
        in_channel=interpolation_in_channel,
        out_channel=interpolation_out_channel,
        num_layers=interpolation_num_layers,
        num_resnet_blocks=interpolation_num_resnet_block,
        attn_layers=interpolation_attn_layers,
        num_heads=interpolation_attn_heads,
        dim_per_head=interpolation_attn_dim_per_head,
        time_dim=interpolation_time_dim,
        cond_dim=interpolation_cond_dim,
        min_channel=interpolation_min_channel,
        max_channel=interpolation_max_channel,
        image_recon=interpolation_img_recon)
    
    interpolation_diffusion_status, interpolation_diffusion_dict= load_checkpoint(interpolation_model_checkpoint)
    if not interpolation_diffusion_status:
        raise Exception("An error occured while loading interpolation model checkpoint!")
    interpolation_diffusion_net.custom_load_state_dict(interpolation_diffusion_dict["model"])
    interpolation_diffusion_net = interpolation_diffusion_net.to(device)
    
    # Generate Interpolation Frames.
    all_frames = None

    # Image Params.
    img_N, img_C, img_F, img_H, img_W = base_video_frames.shape 

    for cond_range in range(0, img_F - 1):
        first_cond_video_frames = base_video_frames[:, :, cond_range, :, :].unsqueeze(2)
        last_cond_video_frames = base_video_frames[:, :, cond_range + 1, :, :].unsqueeze(2)

        noise = torch.randn((img_N, img_C, 1, img_H, img_W), device=device)

        x0_interpolated_frames = frame_interpolation_sampling(
            interpolation_diffusion_net,
            interpolation_noise_degradation,
            noise,
            first_frame=first_cond_video_frames,
            last_frame=last_cond_video_frames,
            min_noise=interpolation_min_noise_step,
            max_noise=interpolation_max_noise_step,
            labels_tensor=None,
            skip_step_size=interpolation_skip_step,
            device=device,
            log=print)
        
        if all_frames is None:
            all_frames = x0_interpolated_frames
        else:
            all_frames = torch.cat((all_frames, x0_interpolated_frames), dim=2)

    return all_frames

def main():
    parser = argparse.ArgumentParser(
        description="Generate Videos using Diffusion Models.")

    parser.add_argument(
        "--seed",
        help="Seed value.",
        default=None,
        type=int)

    parser.add_argument(
        "--out-dir",
        help="File path to save output.",
        default=None,
        type=pathlib.Path)

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

    # Seed Value.
    seed_val = args["seed"]
    if seed_val is not None:
        torch.manual_seed(seed_val)
    
    # Device to run model on.
    device = args["device"]

    # Output Path.
    out_dir = args["out_dir"]

    # Load and Parse config JSON.
    config_json = args["config_path"]
    with open(config_json, 'r') as json_file:
        json_data = json_file.read()
    config_dict = json.loads(json_data)

    x0_frames_approx = generate_base_frames(
        device,
        config_dict)
    x0_frames_approx = generate_interpolated_frames(
        device,
        x0_frames_approx,
        config_dict)

    make_gif(
        x0_frames_approx,
        global_steps=None,
        dest_path=out_dir,
        log=print)

if __name__ == "__main__":
    main()

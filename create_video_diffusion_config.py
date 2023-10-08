import os
import glob
import json

import click

def create_video_diffusion_config():
    config_name = click.prompt(
        "Name of model, will be reflected in json file name?",
        type=str)
    destination_path = click.prompt(
        "Destination path for config file?",
        type=click.Path(exists=True))

    json_file = os.path.join(
        destination_path,
        config_name + ".json")

    json_params = {}

    # Dataset Path: file path to json file.
    json_params["csv_path"] = click.prompt(
        "File path to training dataset (CSV file)?",
        type=click.Path(exists=True))
    
    # Dataset has labels.
    json_params["has_labels"] = click.prompt(
        "Does csv file contain labels?",
        type=bool,
        default=False)
    
    # Conditional Dimension.
    json_params["cond_dim"] = None
    if json_params["has_labels"]:
        json_params["cond_dim"] = click.prompt(
            "Number of labels for each frame?",
            type=click.IntRange(min=1))

    # Output directory for checkpoint.
    json_params["out_dir"] = click.prompt(
        "Destination path for model output?",
        type=click.Path())

    # Load Model checkpoints.
    if click.confirm('Do you want to load a previous model checkpoint?'):
        json_params["model_checkpoint"] = click.prompt(
            "Model checkpoint?",
            type=click.Path(exists=True))
        json_params["load_diffusion_optim"] = click.prompt(
            "Load model's checkpoint optim values?",
            type=bool,
            default=False)
    else:
        json_params["model_checkpoint"] = None
        json_params["load_diffusion_optim"] = False

    if click.confirm('Do you want to load a previous configuration checkpoint?'):
        json_params["config_checkpoint"] = click.prompt(
            "Config chekpoint?",
            type=click.Path(exists=True))
    else:
        json_params["config_checkpoint"] = None
    
    # Super Resolution Training.
    json_params["super_resolution_training"] = False
    if click.confirm('Training SUper Resolution Model (For upsampling)?'):
        json_params["super_resolution_training"] = True
        json_params["low_res_dim"] = click.prompt(
            "Dimensiong for downsampling training image (Low Resolution Dim)?",
            type=click.IntRange(min=2))

    # Frames Window Params.
    json_params["frame_window"] = click.prompt(
        "Total frames to be loaded?",
        type=click.IntRange(min=1),
        default=25)
    
    # Frames Skipped Params.
    json_params["frame_skipped"] = click.prompt(
        "Number of frames to be skipped (Modulo operator is used to select dataset frames)?",
        type=click.IntRange(min=0),
        default=6)

    # Training Parameters.
    json_params["checkpoint_steps"] = click.prompt(
        "Steps to be performed before checkpoint?",
        type=click.IntRange(min=1),
        default=1_000)
    json_params["lr_steps"] = click.prompt(
        "Steps before halving learning rate?",
        type=click.IntRange(min=1),
        default=100_000)
    json_params["max_epoch"] = click.prompt(
        "Total epoch for training?",
        type=click.IntRange(min=1),
        default=1_000)
    json_params["plot_img_count"] = click.prompt(
        "Number of videos to be generated?",
        type=click.IntRange(min=1),
        default=1)

    # Model Params.
    json_params["diffusion_lr"] = click.prompt(
        "Learning Rate for model training?",
        type=click.FloatRange(min=0, min_open=True),
        default=2e-5)
    json_params["batch_size"] = click.prompt(
        "Batch size for training?",
        type=click.IntRange(min=1),
        default=8)

    # Diffusion Params.
    json_params["noise_scheduler"] = click.prompt(
        "Noise scheduler to use?",
        type=click.Choice(["LINEAR", "COSINE"], case_sensitive=False),
        default="LINEAR")

    if json_params["noise_scheduler"] == "LINEAR":
        # Forward process variances used in linear noise scheduling.
        json_params["beta1"] = click.prompt(
            "Beta1 for Linear Noise scheduling?",
            type=click.FloatRange(min=0, min_open=True),
            default=5e-3)
        
        # Forward process variances used in linear noise scheduling.
        json_params["betaT"] = click.prompt(
            "BetaT for Linear Noise scheduling?",
            type=click.FloatRange(min=0, min_open=True),
            default=9e-3)
    else:
        # Default Params just in case changes are made manually.
        json_params["beta1"] = 5e-3
        json_params["betaT"] = 9e-3
    
    json_params["diffusion_alg"] = click.prompt(
        "Diffusion algorithm to use?",
        type=click.Choice(["DDPM", "DDIM", "COLD"], case_sensitive=False),
        default="DDPM")
    
    if json_params["diffusion_alg"] == "DDIM" or json_params["diffusion_alg"] == "COLD":
        json_params["skip_step"] = click.prompt(
            "Number of steps to be skipped in DDIM/COLD sampling?",
            type=click.IntRange(min=1),
            default=100)
    else:
        # Placeholder value just in case json file is manually changed.
        json_params["skip_step"] = 100
    
    json_params["min_noise_step"] = click.prompt(
        "Min noise step for diffusion model?",
        type=click.IntRange(min=1),
        default=1)
    json_params["max_noise_step"] = click.prompt(
        "Max noise step for diffusion model?",
        type=click.IntRange(min=1),
        default=1_000)
    json_params["max_actual_noise_step"] = click.prompt(
        "Max actual noise step, needed for noise scheduler?",
        type=click.IntRange(min=1),
        default=1_000)

    # Model Params.
    json_params["in_channel"] = click.prompt(
        "Model In Channel?",
        type=click.IntRange(min=1),
        default=3)
    json_params["out_channel"] = click.prompt(
        "Model Out Channel?",
        type=click.IntRange(min=1),
        default=3)
    json_params["mapping_channel"] = None
    if json_params["super_resolution_training"]:
        json_params["mapping_channel"] = click.prompt(
            "Mapping Channel?",
            type=click.IntRange(min=2),
            default=128)
    json_params["num_layers"] = click.prompt(
        "Number of layers in model?",
        type=click.IntRange(min=1),
        default=4)
    json_params["num_resnet_block"] = click.prompt(
        "Number of Residual layers in each model's layer?",
        type=click.IntRange(min=1),
        default=1)
    
    # Attention Mechanism for each layer.
    json_params["attn_layers"] = []
    for layer_num in range(json_params["num_layers"]):
        if click.confirm(f"Do you want to add attention mechanism in Layer {layer_num} / {json_params['num_layers'] - 1}?"):
            json_params["attn_layers"].append(layer_num)
    json_params["attn_heads"] = click.prompt(
        "Number of attention heads in attention layers?",
        type=click.IntRange(min=1),
        default=1)
    attn_dim_per_head_val = click.prompt(
        "Dimensions of attention head (-1 for None)?",
        type=click.IntRange(min=-1),
        default=-1)
    json_params["attn_dim_per_head"] = None if attn_dim_per_head_val == -1 else attn_dim_per_head_val
    json_params["time_dim"] = click.prompt(
        "Dimension of time conditional input?",
        type=click.IntRange(min=4),
        default=512)
    json_params["min_channel"] = click.prompt(
        "Minimum channel in model?",
        type=click.IntRange(min=4),
        default=128)
    json_params["max_channel"] = click.prompt(
        "Maximum channel in model?",
        type=click.IntRange(min=4),
        default=512)
    json_params["img_recon"] = click.prompt(
        "Reconstruct frames in final layer (Use Tanh: for cold diffusion)?",
        type=bool,
        default=False)

    try:
        if click.confirm(f"File will be saved in: {json_file}, Are you sure?", default=True):
            with open(json_file, "w") as f:
                json.dump(json_params, f)
            click.echo(f"File saved at: {json_file}")
    except Exception as e:
        click.echo(f"An error occured saving json file: {e}.")

if __name__ == "__main__":
    create_video_diffusion_config()

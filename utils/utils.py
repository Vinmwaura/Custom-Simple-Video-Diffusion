import os

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision


# Print iterations progress
def printProgressBar (
        iteration,
        total,
        prefix='',
        suffix='',
        decimals=1,
        length=100,
        fill='█',
        printEnd="\r",
        log=print):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    log(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        log()

def make_gif(videos, global_steps=None, dest_path=None, log=print):
    for video_index, video in enumerate(videos):
        video_frames = video.permute(1, 0, 2, 3)

        all_frames = []
        for frame in video_frames:
            # Convert Tensor to numpy array.
            image_np = frame.permute(1, 2, 0).cpu().numpy()

            # Clip values to be between -1 and 1 to avoid artefacts.
            image_np = np.clip(image_np, -1, 1)

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Convert to appropriate format.
            image_pil_range = np.uint8(((image_rgb + 1) / 2) * 255)

            # Convert Numpy array to PIL Image.
            pil_image = torchvision.transforms.ToPILImage()(image_pil_range)

            all_frames.append(pil_image)

        if dest_path is None:
            # current_dir = os.path.dirname(os.path.abspath(__file__))
            dir_path = os.path.join(".", "GIFS")
        else:
            dir_path = os.path.join(dest_path, "GIFS")

        os.makedirs(dir_path, exist_ok=True)
        frame_one = all_frames[0]
        try:
            if global_steps is not None and isinstance(global_steps, int):
                path = os.path.join(dir_path, f"video_{video_index}_{global_steps:,}.gif")
            else:
                path = os.path.join(dir_path, f"video_{video_index}.gif")

            frame_one.save(
                path,
                format="GIF",
                append_images=all_frames,
                save_all=True,
                duration=100,
                loop=0)
            log(f"Saving gif: {path}")
        except Exception as e:
            log(f"An error occured while creating gif: {e}")

def plot_sampled_images(sampled_imgs, file_name, dest_path=None, log=print, n_row=5):
    # Convert from BGR to RGB,
    permute = [2, 1, 0]
    sampled_imgs = sampled_imgs[:, permute]

    # TODO: Allow multiple images to be saved separately instead of in a grid.
    grid_img = torchvision.utils.make_grid(
        sampled_imgs,
        nrow=n_row,
        normalize=True,
        value_range=(-1, 1))

    if dest_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.join(current_dir, "plots")
    else:
        dir_path = os.path.join(dest_path, "plots")
    
    os.makedirs(dir_path, exist_ok=True)
    try:
        path = os.path.join(dir_path, str(file_name) + ".jpg")
        torchvision.utils.save_image(
            grid_img,
            path)
        log(f"Saving generated image: {path}")
    except Exception as e:
        log(f"An error occured while plotting reconstructed image: {e}")

def save_model(model_net, file_name, dest_path, checkpoint=False, steps=0, log=print):
    try:
        if checkpoint:
            f_path = os.path.join(dest_path, "checkpoint")
        else:
            f_path = os.path.join(dest_path, "models")
        
        os.makedirs(f_path, exist_ok=True)

        model_name = f"{file_name}_{str(steps)}.pt"
        torch.save(
            model_net,
            os.path.join(f_path, model_name))
        return True
    except Exception as e:
        log(f"Exception occured while saving model: {e}.")
        return False

def load_checkpoint(checkpoint_path, log=print):
    if os.path.exists(checkpoint_path):
        log(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            return True, checkpoint
        except Exception as e:
            return False, None
    else:
        log(f"Checkpoint does not exist.")
        return False, None

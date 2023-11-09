import json
import random
import pathlib
import argparse

import cv2
import numpy as np

import torch
import torch.nn.functional as F

from utils.utils import make_gif
from generate_video import generate_video_diffusion

"""
Mock generate pose keypoints based on either text-prompts or 
other inputs from user input. Assume this function's output was generated 
from a ML model such as an LLM.
"""
def generate_pose_keypoints(f_paths, num_frames=32, skip_frames=2):
    with open(f_paths, "r") as f:
        all_pose_keypoints = json.load(f)
    
    random_index = random.randint(0, len(all_pose_keypoints) - num_frames)
    skip_frames = 1 if skip_frames <= 0 else skip_frames
    selected_pose_keypoints = []
    for frame_index in range(random_index, random_index + num_frames):
        if frame_index % skip_frames == 0:
            selected_pose_keypoints.append(all_pose_keypoints[str(frame_index)])
    return selected_pose_keypoints

"""
Create visualization of pose keypoints as multiple frames.
"""
def generate_frames(pose_keypoints, img_dim=(128, 128, 3)):
    # Indexes of interconnected pose keypoints in order.
    line_indexes = [
        (0,1), (0,2), (1,3), (2,4), (5, 6), (5,7), (7,9), (6, 8), 
        (8, 10), (6,12), (5,11), (11,12), (11,13), (13,15), (12,14), (14,16)]
    
    H, W, C = img_dim

    combined_img = []
    for frame in pose_keypoints:
        img = np.zeros((H, W, C))

        # Adds Circles to each keypoints.
        temp_points = []
        for point in frame:
            x, y = point
            x = int(((x + 1) / 2) * W)
            y = int(((y + 1) / 2) * H)

            img = cv2.circle(
                img,
                (x, y),
                radius=2,
                color=(128,128,128),
                thickness=-1)
            temp_points.append((x, y))

        # Draws line between each keypoints in order.
        for line in line_indexes:
            point1_idx, point2_idx = line

            img = cv2.line(
                img,
                temp_points[point1_idx],
                temp_points[point2_idx],
                (128,0,128),
                thickness=2)
        combined_img.append(img)
    return combined_img

"""
Visualize pose keypoints to be used as labels in models and generate videos.
"""
def visualize_gif(args):
    stop_loop = False

    print("Commands:\
          \nEsc - Quit Program.\
          \nr - Generate new keypoints\
          \ng - Generate videos using model.")

    while not stop_loop:
        selected_pose_keypoints = generate_pose_keypoints(
            f_paths=args["keypoints"],
            num_frames=32,
            skip_frames=2)
        frames_list = generate_frames(selected_pose_keypoints)

        generate_video = False
        loop_images = True
        while loop_images:
            # Display images like a GIF.
            for frame in frames_list:
                cv2.imshow("Frame", frame)

                k = cv2.waitKey(100)

                if k == 27:
                    loop_images = False
                    stop_loop = True
                    break
                elif k == ord("r"):
                    loop_images = False
                    break
                elif k == ord("g"):
                    loop_images = False
                    stop_loop = True
                    generate_video = True
                    break

        cv2.destroyAllWindows()

        if generate_video:
            # Generate video using model.
            try:
                # Save Keypoints frames as gif.
                temp_tensors = None
                for numpy_frames in frames_list:
                    # Scale images to be between 1 and -1.
                    numpy_frames = (numpy_frames.astype(float) - 127.5) / 127.5

                     # Convert image as numpy to Tensor.
                    image_tensor = torch.from_numpy(numpy_frames).float()

                    # Permute image to be of format: [C, H, W]
                    image_tensor = image_tensor.permute(2, 0, 1)

                    # Cases where frames are not square shapes adds padding
                    # i.e (640, 480) => (640, 640).
                    _, H, W = image_tensor.shape
                    diff_ = abs(H - W)
                    image_tensor = F.pad(
                        image_tensor,
                        (0, 0, diff_//2, diff_//2),
                        "constant",
                        -1)
                    
                    # Converts each frame image from 2D to 3D to allow multiple frames to be grouped.
                    _, H, W = image_tensor.shape
                    image_tensor = image_tensor.unsqueeze(1).unsqueeze(0)

                    if temp_tensors is None:
                        temp_tensors = image_tensor
                    else:
                        temp_tensors = torch.cat((temp_tensors, image_tensor), dim=2)

                make_gif(
                    temp_tensors,
                    global_steps=0,
                    dest_path=args["out_dir"],
                    log=print)

                commands = [
                    "--config",
                    str(args["config"])]
                
                commands.append("--out-dir")
                commands.append(str(args["out_dir"]))
                
                commands.append("--device")
                commands.append(args["device"])
                
                commands.append("--label")
                for keypoint in selected_pose_keypoints:
                    for x_keypoint, y_keypoint in keypoint:
                        commands.append(str(x_keypoint))
                        commands.append(str(y_keypoint))
                generate_video_diffusion(raw_args=commands)

            except Exception as e:
                print(f"An error occured generating model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pose Keypoints Visualization and Video Generation.")
    parser.add_argument(
        "-k",
        "--keypoints",
        help="File path to pose keypoints file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "-c",
        "--config",
        help="File path to Model(s) config file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-dir",
        help="File path to save output.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--device",
        help="Which hardware device will model run on (default='cpu')?",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")

    args = vars(parser.parse_args())
    visualize_gif(args)

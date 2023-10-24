import os
import json
import glob
import random

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


"""
Custom Loader for Videos Frames using Opencv2.
"""
class VideoFramesDataset(Dataset):
    def __init__(
            self,
            json_path,
            load_labels=False,
            load_cond_images=False,
            frame_window=24,
            frames_skipped=0):

        self.frame_dict = {}

        self.load_labels = load_labels
        self.load_cond_images = load_cond_images

        # Dataset_src is a csv with labels e.g one-hot encodings.
        if not os.path.isfile(json_path):
            raise Exception("Invalid / No json file path found.")

        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
        
        for data in json_data["data"]:
            frame_num = data["image_path"].split("/")[-1].split(".")[0]
            self.frame_dict[frame_num] = data

        # Number of sequential frames loaded together.
        if frame_window < 1 or frame_window >= len(self.frame_dict):
            raise ValueError("Invalid frame window value.")

        self.frame_window = frame_window

        # Frames skipped in frame window.
        if frames_skipped < 0 or frames_skipped >= frame_window:
            raise ValueError("Invalid frame skipped value.")

        self.frames_skipped = (frames_skipped + 1)

        self.frame_counts = len(self.frame_dict)

    def __len__(self):
        return self.frame_counts - self.frame_window
    
    def __getitem__(self, index):
        labels = []
        combined_image = None
        combined_cond_image = None

        for frame_count in range(0, self.frame_window):
            if frame_count % self.frames_skipped == 0:
                frame_index = index + frame_count

                # Combines all label vectors together to form a larger vector.
                if self.load_labels and "labels" in self.frame_dict[str(frame_index)]:
                    labels.extend(self.frame_dict[str(frame_index)]["labels"])

                image_path = self.frame_dict[str(frame_index)]["image_path"]

                # Load images using opencv2.
                image = cv2.imread(image_path)

                # Scale images to be between 1 and -1.
                image = (image.astype(float) - 127.5) / 127.5

                # Convert image as numpy to Tensor.
                image_tensor = torch.from_numpy(image).float()

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
                image_tensor = image_tensor.unsqueeze(1)

                # Groups frames together to 3D input.
                if combined_image is None:
                    combined_image = image_tensor
                else:
                    combined_image = torch.cat(
                        (combined_image, image_tensor),
                        dim=1)
                
                # Conditional images such as Low-Resolution images or Segmentations Masks.
                if self.load_cond_images:
                    cond_image_path = self.frame_dict[str(frame_index)]["cond_image_path"]

                    # Load images using opencv2.
                    cond_image = cv2.imread(cond_image_path)

                    # Scale images to be between 1 and -1.
                    cond_image = (cond_image.astype(float) - 127.5) / 127.5

                    # Convert image as numpy to Tensor.
                    cond_image_tensor = torch.from_numpy(cond_image).float()

                    # Permute image to be of format: [C, H, W]
                    cond_image_tensor = cond_image_tensor.permute(2, 0, 1)

                    # Cases where frames are not square shapes adds padding
                    # i.e (640, 480) => (640, 640).
                    _, H, W = cond_image_tensor.shape
                    diff_ = abs(H - W)
                    cond_image_tensor = F.pad(
                        cond_image_tensor,
                        (0, 0, diff_//2, diff_//2),
                        "constant",
                        -1)
                    
                    # Converts each frame image from 2D to 3D to allow multiple frames to be grouped.
                    _, H, W = cond_image_tensor.shape
                    cond_image_tensor = cond_image_tensor.unsqueeze(1)

                    # Groups frames together to 3D input.
                    if combined_cond_image is None:
                        combined_cond_image = cond_image_tensor
                    else:
                        combined_cond_image = torch.cat(
                            (combined_cond_image, cond_image_tensor),
                            dim=1)

        labels_tensor = torch.tensor(labels)

        if not self.load_labels and not self.load_cond_images:
            return combined_image
        elif self.load_labels and not self.load_cond_images:
            return combined_image, labels_tensor
        elif not self.load_labels and self.load_cond_images:
            return combined_image, combined_cond_image
        else:
            return combined_image, combined_cond_image, labels_tensor

import os
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
            img_regex,
            frame_window=24,
            frames_skipped=6):
        img_paths = glob.glob(img_regex)
        
        # Number of sequential frames loaded together.
        self.frame_window = frame_window

        # Frames skipped in frame window using modulo operator(%).
        self.frames_skipped = frames_skipped

        self.frame_counts = len(img_paths)
        print(self.frame_counts)

        self.frame_dict = {}
        for img_path in img_paths:
            frame_num = img_path.split("/")[-1].split(".")[0]
            self.frame_dict[frame_num] = img_path

    def __len__(self):
        return self.frame_counts - self.frame_window
    
    def __getitem__(self, index):
        combined_image = None

        for frame_count in range(0, self.frame_window):
            if frame_count % self.frames_skipped == 0:
                frame_index = index + frame_count

                img_path = self.frame_dict[str(frame_index)]

                # Load images using opencv2.
                img = cv2.imread(img_path)

                # Scale images to be between 1 and -1.
                img = (img.astype(float) - 127.5) / 127.5

                # Convert image as numpy to Tensor.
                img_tensor = torch.from_numpy(img).float()

                # Permute image to be of format: [C, H, W]
                img_tensor = img_tensor.permute(2, 0, 1)

                # Cases where frames are not square shapes adds padding
                # i.e (640, 480) => (640, 640).
                _, H, W = img_tensor.shape
                diff_ = abs(H - W)
                img_tensor = F.pad(img_tensor, (0, 0, diff_//2, diff_//2), "constant", -1)
                
                # Converts each frame image from 2D to 3D to allow multiple frames to be grouped.
                _, H, W = img_tensor.shape
                img_tensor = img_tensor.unsqueeze(1)

                # Groups frames together to 3D input.
                if combined_image is None:
                    combined_image = img_tensor
                else:
                    combined_image = torch.cat((combined_image, img_tensor), dim=1)

        return combined_image

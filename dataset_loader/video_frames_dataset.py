import os
import csv
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
            csv_path,
            frame_window=24,
            frames_skipped=0):
        self.frame_dict = {}

        # Dataset_src is a csv with labels e.g one-hot encodings.
        if not os.path.isfile(csv_path):
            raise Exception("Invalid / No csv file path found.")
        
        count = 0
        with open(csv_path, "r") as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip Header (Assumes it exists).

            for row in reader:
                count += 1

                img_path = row[0]
                labels = [float(label) for label in row[1:]]

                frame_num = img_path.split("/")[-1].split(".")[0]
                self.frame_dict[frame_num] = {
                    "img_path": img_path,
                    "labels": labels}

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

        for frame_count in range(0, self.frame_window):
            if frame_count % self.frames_skipped == 0:
                frame_index = index + frame_count

                # Combines all label vectors together to form a larger vector.
                if "labels" in self.frame_dict[str(frame_index)]:
                    labels.extend(self.frame_dict[str(frame_index)]["labels"])

                img_path = self.frame_dict[str(frame_index)]["img_path"]

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

        labels_tensor = torch.tensor(labels)
        return combined_image, labels_tensor

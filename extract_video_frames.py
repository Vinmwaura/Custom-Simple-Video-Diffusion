import os
import pathlib
import argparse

import cv2
import numpy as np

def extract_frames(
        video_path,
        dest_path,
        max_file_count=1_000,
        resize_dim=64):
    
    frame_count = 0
    folder_count = 0

    cap = cv2.VideoCapture(str(video_path))

    while cap.isOpened():
        if frame_count > 0 and frame_count % max_file_count == 0:
            folder_count = folder_count + 1

        folder_path = os.path.join(str(dest_path), str(folder_count))
        os.makedirs(folder_path, exist_ok=True)

        _, frame = cap.read()

        if frame is not None:
            H, W, _ = frame.shape

            # Pad images with black pixels to be a square dimension.
            if W > H:
                pad_val = (W - H) // 2
            else:
                pad_val = (H - W) // 2

            frame = cv2.copyMakeBorder(
                frame,
                top=pad_val,
                bottom=pad_val,
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT)

            resized_frame = cv2.resize(frame, (resize_dim, resize_dim))
            cv2.imwrite(
                os.path.join(folder_path, "%d.jpg" % frame_count),
                resized_frame)
            
            frame_count = frame_count + 1
        
        else:
            break

        print(f"Frames: {frame_count:,} - Dataset Source: {video_path}")

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()

def main():
    parser = argparse.ArgumentParser(
        description="Generate Video Dataset.")

    parser.add_argument(
        "-v",
        "--video-path",
        help="File path to video.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--dest-path",
        help="File path to save output.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--resize-dim",
        help="Dimension to resize image(Default: 64)",
        default=64,
        type=int)
    parser.add_argument(
        "--max-file-count",
        help="Max file in each folder.",
        default=1000,
        type=int)

    args = vars(parser.parse_args())

    video_path = args["video_path"]
    dest_path = args["dest_path"]
    resize_dim = args["resize_dim"]
    max_file_count = args["max_file_count"]

    extract_frames(
        video_path=video_path,
        dest_path=dest_path,
        max_file_count=max_file_count,
        resize_dim=resize_dim)

if __name__ == "__main__":
    main()

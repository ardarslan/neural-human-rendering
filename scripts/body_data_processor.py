import os
import numpy as np
import cv2
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments for preprocessing script")

    parser.add_argument(
        "--videos_dir",
        type=str,
        help="Directory where the videos are located.",
        required=True,
    )
    parser.add_argument(
        "--target_h",
        type=int,
        help="Target height of body images.",
        default=224,
    )
    parser.add_argument(
        "--target_w",
        type=int,
        help="Target width of body images.",
        default=112,
    )

    args = parser.parse_args()

    for split in ["train", "validation", "test"]:
        for data_type in ["input", "output"]:
            current_files_dir = os.path.join(args.videos_dir, split, data_type)
            os.makedirs(current_files_dir.replace("_temp", ""), exist_ok=True)
            file_paths = [
                os.path.join(current_files_dir, file_name)
                for file_name in os.listdir(current_files_dir)
                if file_name[-4:] == ".png"
            ]
            for file_path in file_paths:
                current_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                resized = cv2.resize(
                    current_image,
                    (args.target_w, args.target_h),
                    interpolation=cv2.INTER_AREA,
                )
                padded = np.ones((args.target_h, args.target_h, 3)) * 255
                padding = int((args.target_h - args.target_w) / 2)
                padded[:, padding : padding + args.target_w, :] = resized
                cv2.imwrite(file_path.replace("_temp", ""), padded)

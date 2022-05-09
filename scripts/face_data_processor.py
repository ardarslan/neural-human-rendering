import os
import cv2
import argparse
from pathlib import Path
from face_data_utils import extract_face_edge_map_from_single_image

save_frame_idx = 0


def extract_inputs_and_outputs_for_one_video(
    video_path, target_h_w, skip_frame, use_canny_edges
):
    global save_frame_idx
    videos_dir = str(Path(video_path).parent.parent)

    inputs_dir = os.path.join(videos_dir, "input")
    os.makedirs(os.path.join(inputs_dir), exist_ok=True)

    outputs_dir = os.path.join(videos_dir, "output")
    os.makedirs(os.path.join(outputs_dir), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    read_frame_idx = 0
    while True:
        success, image = cap.read()

        # End of the video
        if not success:
            break

        # Read only "skip_frame"th frame
        if read_frame_idx % skip_frame != 0:
            read_frame_idx += 1
            continue

        try:
            (
                cropped_and_resized_image,
                face_edge_map,
            ) = extract_face_edge_map_from_single_image(
                image, target_h_w, use_canny_edges
            )
        except Exception as e:
            print(e)
            read_frame_idx += 1
            continue

        # Write cropped_and_resized_image to disk
        cv2.imwrite(
            os.path.join(outputs_dir, str(save_frame_idx).zfill(7)) + ".png",
            cropped_and_resized_image,
        )

        # Write face_edge_map to disk
        cv2.imwrite(
            os.path.join(inputs_dir, str(save_frame_idx).zfill(7)) + ".png",
            face_edge_map,
        )

        read_frame_idx += 1
        save_frame_idx += 1


def extract_frames_and_edges_for_all_videos(
    videos_dir, target_h_w, skip_frame, split, use_canny_edges
):
    current_videos_dir = os.path.join(videos_dir, split, "original")
    video_paths = [
        os.path.join(videos_dir, split, "original", video_name)
        for video_name in sorted(os.listdir(current_videos_dir))
        if len(video_name) >= 4 and video_name[-4:] == ".avi"
    ]

    for video_path in video_paths:
        extract_inputs_and_outputs_for_one_video(
            video_path, target_h_w, skip_frame, use_canny_edges
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for preprocessing script")

    parser.add_argument(
        "--videos_dir",
        type=str,
        help="Directory where the videos are located.",
        required=True,
    )
    parser.add_argument(
        "--use_canny_edges",
        type=str,
        required=True,
        help="Whether to use canny edges or not.",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Split should be 'train', 'test' or 'validation'.",
        required=True,
    )
    parser.add_argument(
        "--target_h_w",
        type=int,
        help="Target width and height of face edge map.",
        default=256,
    )
    parser.add_argument(
        "--skip_frame",
        type=int,
        help="Number of frames to skip while reading.",
        default=1,
    )

    args = parser.parse_args()

    extract_frames_and_edges_for_all_videos(
        args.videos_dir,
        args.target_h_w,
        args.skip_frame,
        args.split,
        args.use_canny_edges,
    )

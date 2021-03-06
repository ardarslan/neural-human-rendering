#!/bin/bash

 while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        declare $v
   fi
  shift
 done

mkdir $DATASETS_DIR
VIDEO_DIR=$DATASETS_DIR/face_reconstruction
mkdir $VIDEO_DIR

wget -O $DATASETS_DIR/face_reconstruction_video.mp4 "https://www.dropbox.com/s/htjtuhgpewgcmlh/videoplayback.mp4?dl=1"

mkdir $VIDEO_DIR/train
mkdir $VIDEO_DIR/train/input
mkdir $VIDEO_DIR/train/output
mkdir $VIDEO_DIR/train/original

mv $DATASETS_DIR/face_reconstruction_video.mp4 $VIDEO_DIR/train/original

# Process the whole video
bsub -n 4 -W 24:00 -R "rusage[mem=8192]" -o train.txt python face_data_processor.py --videos_dir $VIDEO_DIR --use_canny_edges $USE_CANNY_EDGES --split train

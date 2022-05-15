#!/bin/bash

# Read arguments and map to respective variables
 while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        declare $v
   fi
  shift
 done

mkdir $DATASETS_DIR
VIDEO_DIR=$DATASETS_DIR/face_video
mkdir $VIDEO_DIR

wget -O $DATASETS_DIR/face_reconstruction_video.mp4 "https://www.dropbox.com/s/htjtuhgpewgcmlh/videoplayback.mp4?dl=1"

mkdir $VIDEO_DIR/validation; mkdir $VIDEO_DIR/validation/original; mv $VIDEO_DIR/val/original/* $VIDEO_DIR/validation/original;

bsub -n 1 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python face_data_processor.py --videos_dir $VIDEO_DIR --use_canny_edges $USE_CANNY_EDGES --split train
bsub -n 1 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python face_data_processor.py --videos_dir $VIDEO_DIR --use_canny_edges $USE_CANNY_EDGES --split validation
bsub -n 1 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python face_data_processor.py --videos_dir $VIDEO_DIR --use_canny_edges $USE_CANNY_EDGES --split test
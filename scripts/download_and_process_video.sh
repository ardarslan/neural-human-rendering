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
bsub -n 1 -W 24:00 -R "rusage[mem=8192]" -o train.txt python face_data_processor.py --videos_dir $VIDEO_DIR --use_canny_edges $USE_CANNY_EDGES --split train

# Create the data folders
mkdir $VIDEO_DIR/test
mkdir $VIDEO_DIR/test/input
mkdir $VIDEO_DIR/test/output
mkdir $VIDEO_DIR/validation
mkdir $VIDEO_DIR/validation/input
mkdir $VIDEO_DIR/validation/output

# Separate the data for input
cd $VIDEO_DIR/train/input
mv `ls | tail -75000` "../../test/input"
mv `ls | tail -100` "../../validation/input"

cd ../../test/input
i=0
for file in *.png; do
    number="000000${i}"
    mv "$file" "${number: -7}.png"
    ((i=i+1))
done

cd ../../validation/input
i=0
for file in *.png; do
    number="000000${i}"
    mv "$file" "${number: -7}.png"
    ((i=i+1))
done


# Separate the data for output
cd $VIDEO_DIR/train/output
mv `ls | tail -75000` "../../test/output"
mv `ls | tail -100` "../../validation/output"

cd ../../test/output
i=0
for file in *.png; do
    number="000000${i}"
    mv "$file" "${number: -7}.png"
    ((i=i+1))
done

cd ../../validation/output
i=0
for file in *.png; do
    number="000000${i}"
    mv "$file" "${number: -7}.png"
    ((i=i+1))
done
#!/bin/bash

 while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        declare $v
   fi
  shift
 done

VIDEO_DIR=$DATASETS_DIR/face_reconstruction

# Create the data folders
mkdir $VIDEO_DIR/test
mkdir $VIDEO_DIR/test/input
mkdir $VIDEO_DIR/test/output
mkdir $VIDEO_DIR/validation
mkdir $VIDEO_DIR/validation/input
mkdir $VIDEO_DIR/validation/output

# Separate the data for input
cd $VIDEO_DIR/train/input
mv `ls | tail -${TEST_SEP}` "${VIDEO_DIR}/test/input"
mv `ls | tail -${VAL_SEP}` "${VIDEO_DIR}/validation/input"

cd $VIDEO_DIR/test/input
i=0
for file in *.png; do
    number="000000${i}"
    mv "$file" "${number: -7}.png"
    ((i=i+1))
done

cd $VIDEO_DIR/validation/input
i=0
for file in *.png; do
    number="000000${i}"
    mv "$file" "${number: -7}.png"
    ((i=i+1))
done


# Separate the data for output
cd $VIDEO_DIR/train/output
mv `ls | tail -75000` "${VIDEO_DIR}/test/output"
mv `ls | tail -100` "${VIDEO_DIR}/validation/output"

cd $VIDEO_DIR/test/output
i=0
for file in *.png; do
    number="000000${i}"
    mv "$file" "${number: -7}.png"
    ((i=i+1))
done

cd $VIDEO_DIR/validation/output
i=0
for file in *.png; do
    number="000000${i}"
    mv "$file" "${number: -7}.png"
    ((i=i+1))
done
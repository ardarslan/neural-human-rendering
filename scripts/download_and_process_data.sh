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
FACE_DIR=$DATASETS_DIR/face
mkdir $FACE_DIR

python3 face_data_downloader.py -d compressed -o $DATASETS_DIR --not_altered --not_mask
mv $DATASETS_DIR/FaceForensics_compressed/* $FACE_DIR;
# rm -rf $DATASETS_DIR/FaceForensics_compressed
mkdir $FACE_DIR/validation; mkdir $FACE_DIR/validation/original; mv $FACE_DIR/val/original/* $FACE_DIR/validation/original;
# rm -rf $FACE_DIR/val
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python face_data_processor.py --videos_dir $FACE_DIR --use_canny_edges $USE_CANNY_EDGES --split train
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python face_data_processor.py --videos_dir $FACE_DIR --use_canny_edges $USE_CANNY_EDGES --split validation
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python face_data_processor.py --videos_dir $FACE_DIR --use_canny_edges $USE_CANNY_EDGES --split test

# wget -O $DATASETS_DIR/body.zip "https://www.dropbox.com/s/coapl05ahqalh09/smplpix_data_test_final.zip?dl=1"
# unzip $DATASETS_DIR/body.zip -d $DATASETS_DIR/body_smplpix_temp
# mv $DATASETS_DIR/body_smplpix_temp/smplpix_data/* $DATASETS_DIR/body_smplpix_temp/
# rm -rf $DATASETS_DIR/body_smplpix_temp/smplpix_data
# rm -rf $DATASETS_DIR/body.zip
# rm -rf $DATASETS_DIR/body_smplpix_temp/validation/input/*; rm -rf $DATASETS_DIR/body_smplpix_temp/validation/output/*
# mv $DATASETS_DIR/body_smplpix_temp/train/input/009* $DATASETS_DIR/body_smplpix_temp/validation/input/
# mv $DATASETS_DIR/body_smplpix_temp/train/input/01* $DATASETS_DIR/body_smplpix_temp/validation/input/
# mv $DATASETS_DIR/body_smplpix_temp/train/output/009* $DATASETS_DIR/body_smplpix_temp/validation/output/
# mv $DATASETS_DIR/body_smplpix_temp/train/output/01* $DATASETS_DIR/body_smplpix_temp/validation/output/
# python3 body_data_processor.py --videos_dir $DATASETS_DIR/body_smplpix_temp
# rm -rf $DATASETS_DIR/body_smplpix_temp

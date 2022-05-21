# Virtual Humans Course Project: Vision Transformers for Neural Human Rendering

## Install Miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Close the current terminal and open a new one.

## Setup Conda Environment

```
conda env create -f environment.yml
```

## Load Modules

Before loading the modules always make sure that you are not in any conda environment. (Even the "base")
```
conda deactivate
conda deactivate
module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy
```

## Activate Conda Environment

```
conda activate virtual_humans
```

## Download and Process Data
### Faceforensics

```
cd scripts
chmod +x download_and_process_data.sh
./download_and_process_data.sh "--DATASETS_DIR=/path/to/data/directory" "--USE_CANNY_EDGES=True"
```
### Face_reconstruction
```
cd scripts
chmod +x download_and_process_video.sh
./download_and_process_video.sh "--DATASETS_DIR=/path/to/data/directory" "--USE_CANNY_EDGES=True"
# Wait for the script and its corresponding jobs to finish
chmod +x separate.sh
./separate.sh "--DATASETS_DIR=/path/to/data/directory" "--TEST_SEP=20000" "--VAL_SEP=100"
```

## Run experiments

```
cd src
```

If you want to keep training using a previous checkpoint use --experiment_time TIMESTAMP_OF_PREVIOUS_TRAIN_JOB

### Train Original Pix2Pix

```
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python train.py --datasets_dir /cluster/scratch/aarslan/virtual_humans_data --dataset_type face --discriminator_type cnn --checkpoints_dir /cluster/scratch/aarslan/virtual_humans_checkpoints --generator_lr 0.0002 --discriminator_lr 0.000065
```

### Train VIT Pix2Pix

```
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python train.py --datasets_dir /path/to/data/directory --dataset_type face --discriminator_type vit --vanilla --projection_dim 32 --num_heads 2 --num_transformer_layers 3 --checkpoints_dir /path/to/checkpoints/directory --generator_lr GEN_LR --discriminator_lr DISC_LR
```

### Train CLIP Pix2Pix

```
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python train.py --discriminator_type clip --clip_fine_tune --dataset_type face --datasets_dir /path/to/data/directory --checkpoints_dir /path/to/checkpoints/directory --generator_lr GEN_LR --discriminator_lr DISC_LR
```



### Test Original Pix2Pix

```
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python test.py --datasets_dir /path/to/data/directory --dataset_type DATASET_TYPE --discriminator_type cnn --checkpoints_dir /path/to/checkpoints/directory --experiment_time TIMESTAMP_OF_TRAIN_JOB
```

### Test VIT Pix2Pix

```
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python test.py --datasets_dir /path/to/data/directory --dataset_type DATASET_TYPE --discriminator_type vit --vanilla --projection_dim 32 --num_heads 2 --num_transformer_layers 3 --checkpoints_dir /path/to/checkpoints/directory --experiment_time TIMESTAMP_OF_TRAIN_JOB
```

### Test CLIP Pix2Pix

```
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python test.py --datasets_dir /path/to/data/directory --dataset_type DATASET_TYPE --discriminator_type clip --checkpoints_dir /path/to/checkpoints/directory --experiment_time TIMESTAMP_OF_TRAIN_JOB
```

### Evaluate results (for any model)

```
bsub -n 2 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python evaluation_metrics.py --datasets_dir /path/to/data/directory --dataset_type DATASET_TYPE --checkpoints_dir /path/to/checkpoints/directory --experiment_time TIMESTAMP_OF_TRAIN_JOB --fid_device cuda:0
```
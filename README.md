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

## Activate Conda Environment

```
conda activate virtual_humans
```

## Download and Process Data

```
cd scripts
chmod +x download_and_process_data.sh
./download_and_process_data.sh "--DATASETS_DIR=/path/to/data/directory" "--USE_CANNY_EDGES=True"
```

## Run experiments

```
cd src
module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy
```

If you want to keep training using a previous checkpoint use --experiment_time TIMESTAMP_OF_PREVIOUS_TRAIN_JOB

### Train Original Pix2Pix on Face dataset

```
bsub -n 4 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=32768]" python train.py --datasets_dir /path/to/data/directory --dataset_type face --discriminator_type cnn --checkpoints_dir /path/to/checkpoints/directory
```

### Train Original Pix2Pix on Body dataset

```
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py --datasets_dir /path/to/data/directory --dataset_type body_smplpix --discriminator_type cnn --checkpoints_dir /path/to/checkpoints/directory
```

### Train VIT Pix2Pix on Face dataset

```
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py --datasets_dir /path/to/data/directory --dataset_type face --discriminator_type vit --vanilla --projection_dim 32 --num_heads 2 --num_transformer_layers 3 --checkpoints_dir /path/to/checkpoints/directory
```

### Train VIT Pix2Pix on Body dataset

```
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py --datasets_dir /path/to/data/directory --dataset_type body_smplpix --discriminator_type vit --vanilla --projection_dim 32 --num_heads 2 --num_transformer_layers 3 --checkpoints_dir /path/to/checkpoints/directory
```




### Test Original Pix2Pix on Face dataset

```
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python test.py --datasets_dir /path/to/data/directory --dataset_type face --discriminator_type cnn --checkpoints_dir /path/to/checkpoints/directory --experiment_time TIMESTAMP_OF_TRAIN_JOB
```

### Test Original Pix2Pix on Body dataset

```
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python test.py --datasets_dir /path/to/data/directory --dataset_type body_smplpix --discriminator_type cnn --checkpoints_dir /path/to/checkpoints/directory --experiment_time TIMESTAMP_OF_TRAIN_JOB
```

### Test VIT Pix2Pix on Face dataset

```
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python test.py --datasets_dir /path/to/data/directory --dataset_type face --discriminator_type vit --vanilla --projection_dim 32 --num_heads 2 --num_transformer_layers 3 --checkpoints_dir /path/to/checkpoints/directory --experiment_time TIMESTAMP_OF_TRAIN_JOB
```

### Test VIT Pix2Pix on Body dataset

```
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python test.py --datasets_dir /path/to/data/directory --dataset_type body_smplpix --discriminator_type vit --vanilla --projection_dim 32 --num_heads 2 --num_transformer_layers 3 --checkpoints_dir /path/to/checkpoints/directory --experiment_time TIMESTAMP_OF_TRAIN_JOB
```

### Evaluate results on Face dataset (for any model)

```
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python evaluation_metrics.py --datasets_dir /path/to/data/directory --dataset_type face --checkpoints_dir /path/to/checkpoints/directory --experiment_time TIMESTAMP_OF_TRAIN_JOB --fid_device cuda:0
```
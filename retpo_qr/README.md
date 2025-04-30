<!-- <p align="center">
    🤗 <a href="https://huggingface.co/collections/alignment-handbook/handbook-v01-models-and-datasets-654e424d22e6880da5ebc015" target="_blank">Datasets</a> | 🤗 <a href="https://huggingface.co/collections/alignment-handbook/handbook-v01-models-and-datasets-654e424d22e6880da5ebc015" target="_blank">Models</a>
</p> -->

# RetPO - Retrievers' Preference Optimization for LLMs

Recipes for fine-tuning LLMs with retriever preferences.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Models](#models)

## Features
- Supervised Fine-tuning (SFT) with retrieval enhancement
- Direct Preference Optimization (DPO) for further alignment
- Support for multiple datasets (QReCC, TopiOCQA)
- Distributed training with DeepSpeed integration

## Requirements
- Python 3.10+
- CUDA-compatible GPU(s)
- Minimum 40GB GPU RAM (80GB recommended for full-scale training)
- Git LFS
- Hugging Face account

## Installation

### 1. Environment Setup
```shell
# Create and activate conda environment
conda create -n retpo_qr python=3.10
conda activate retpo_qr

# Install PyTorch 2.1.0
# Visit https://pytorch.org/get-started/locally/ for specific installation commands
```

### 2. Install Dependencies
```shell
# Install main packages
python -m pip install .

# Install Flash Attention 2 (Optional, recommended for faster training)
# For machines with <96GB RAM, reduce MAX_JOBS
python -m pip install flash-attn --no-build-isolation

# Install Git LFS
sudo apt-get install git-lfs

# Login to Hugging Face
huggingface-cli login
```

### Download Datasets

To download the `qr_dataset` directory from the Hugging Face repository, use the following command:

```shell
huggingface-cli download dmis-lab/RF-Collection \
  --repo-type dataset \
  --include "qr_dataset/*" \
  --local-dir ./dataset
```

This will download the entire `qr_dataset` directory and save it to the `./dataset` directory in your current working directory

## Usage

### Training
You can now check out the `recipes` directory for configuration for training models!

#### Single GPU Training

For Supervised Fine-tuning (SFT):

```shell
python run_cqr.py recipes/sft/config_sft.yaml
```

For Direct Preference Optimization (DPO):
```shell
python run_cqr_dpo.py recipes/dpo/config_dpo.yaml
```

#### Multi-GPU Training
It would use 8 gpus in 
default with `accelerate` library. We used 8 A100 
GPUs with 80GB memory. you may want to modify the 
batch size in each recipe to adjust the memory 
consumption.
1. Configure `accelerate`:
```shell
accelerate config
```

2. Launch training:
```shell
export SCRIPT_PATH='run_cqr.py'  # or run_cqr_dpo.py
export RECIPE_PATH='recipes/sft/config_sft.yaml'  # or recipes/dpo/config_dpo.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    $SCRIPT_PATH $RECIPE_PATH
```

### Inference

Generate rewrites using trained models:
```shell
export DATASET_NAME='qrecc'  # or 'topiocqa'
export MODEL_PATH='path/to/your/checkpoint'
export BATCH_SIZE=8  # adjust based on GPU memory
export ARGS=""

python inf_cqr.py \
    --do_eval true \
    --dataset_dir dataset/$DATASET_NAME/bm25/test.json \
    --output_dir $MODEL_PATH \
    --model_name_or_path $MODEL_PATH \
    --per_device_eval_batch_size $BATCH_SIZE \
    --eval_split test \
    $ARGS
```

## Models

Pre-trained models are available on the Hugging Face Hub. Instructions for downloading and using specific models will be added soon.


## Acknowledgments
This project is inspired by and builds upon the [Alignment Handbook](https://github.com/huggingface/alignment-handbook) repository.
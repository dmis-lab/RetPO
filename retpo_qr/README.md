<!-- <p align="center">
    🤗 <a href="https://huggingface.co/collections/alignment-handbook/handbook-v01-models-and-datasets-654e424d22e6880da5ebc015" target="_blank">Datasets</a> | 🤗 <a href="https://huggingface.co/collections/alignment-handbook/handbook-v01-models-and-datasets-654e424d22e6880da5ebc015" target="_blank">Models</a>
</p> -->

# RetPO - LLM Alignment

Recipes for fine-tuning LLMs with retriever preferences.


---

## Installation Instructions  

### Step 1: Create a Virtual Environment  
First, create a Python virtual environment using Conda:  

```shell
conda create -n retpo_qr python=3.10 && conda activate retpo_qr
```

## Step 2: Install PyTorch (Hardware-Dependent)

Next, install PyTorch v2.1.0 (this specific version is crucial for reproducibility).
Follow the instructions on the PyTorch Installation Page based on your hardware.

## Step 3: Install Dependencies

Once PyTorch is installed, navigate to the project directory and install the required dependencies:

```shell
python -m pip install .
```

Step 4: Install Flash Attention 2 (Optional, but Recommended for Faster Training)

If your machine has less than 96GB of RAM and many CPU cores, reduce MAX_JOBS, e.g.:

```shell
python -m pip install flash-attn --no-build-isolation
```

### Step 5: Log in to Hugging Face

Authenticate your Hugging Face account to access model repositories:
```shell
huggingface-cli login
```

### Step 6: Install Git LFS

Git LFS is required for handling large model files. Install it using:

```shell
sudo apt-get install git-lfs
```

You can now check out the `recipes` directory for configuration for training models!

### Download Datasets

To use the datasets, download them from Hugging Face Hub using the following command:


## Running the Model

### Train with a Single GPU

Use the following command to train on a single GPU:

```shell
python run_cqr.py recipes/
```

### Train with Multiple GPUs

Before running the scripts on multiple GPUs, configure accelerate:

```shell
accelerate config
```

### Train models with a single command

We train a model in pipeline manner fine-tune LLM to answer the question first and then train it to rewrite the questions. To follow it, you can run the following command. It would use 8 gpus in default with `accelerate` library. We used 8 A100 GPUs with 80GB memory. you may want to modify the batch size in each recipe to adjust the memory consumption.

Then, run the training script:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml run_cqr_dpo.py recipes/llama2-7b/qrecc/dpo/oqf-bm25/config_full_multi.yaml
```

### Train Models in a Pipeline (End-to-End)

We train models in a pipeline manner, first fine-tuning an LLM to answer questions and then training it to rewrite queries.
To follow this approach, run:

```shell
bash train_cqr.sh
```

This setup defaults to 8 GPUs using accelerate. We used 8 A100 GPUs (80GB memory),
but you may need to adjust the batch size in each recipe based on available memory.

### Inference (Generating Model Outputs)

To generate inference results from a trained model, use:


```shell
export MODEL_NAME="results/llama2-7b/oqf-joint-bm25/dpo/one_pair"
export GPU=1

export MODEL_PATH=$MODEL_NAME
export OUTPUT_PATH=$MODEL_NAME
export ARGS=""
echo model_name-$MODEL_NAME
echo gpu-$GPU

CUDA_VISIBLE_DEVICES=$GPU python inf_cqr.py \
    --do_eval true \
    --dataset_dir dataset/qrecc-analysis \
    --output_dir $OUTPUT_PATH \
    --do_eval true \
    --model_name_or_path $MODEL_PATH \
    --per_device_eval_batch_size 2 \
    --eval_split eval-100 \
    $ARGS
```

### Download Our Models

To use our pre-trained models, download them from Hugging Face Hub using:


### Reference

This repository is inspired by and follows best practices from the Alignment Handbook repository.
For further details, check the original repository: link
#!/bin/bash

# Define variables for the script
ROOT_DIR="/mnt/madehua/fooddata/Food2k_complete"
TRAIN_FILE="train.txt"
TEST_FILE="test.txt"
BATCH_SIZE=256
LEARNING_RATE=0.001
EPOCHS=10
NUM_CLASSES=2000
SAVE_DIR="/mnt/madehua/model/checkpoints/dinov2/food2k"
CONFIG_FILE="dinov2/configs/ssl_food_vitl14.yaml"  # Path to your YAML configuration file
# Number of GPUs to use
NUM_GPUS=4  # Adjust this to the number of GPUs you have
export OMP_NUM_THREADS=4  # Adjust based on your CPU cores
#export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355  # Choose an available port
# export NCCL_IB_TIMEOUT=22
# Run the Python script with the specified arguments using DDP
# Run the Python script with the specified arguments using DDP
torchrun --nproc_per_node=$NUM_GPUS finetune/finetune.py \
    --root_dir $ROOT_DIR \
    --train_file $TRAIN_FILE \
    --test_file $TEST_FILE \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --num_classes $NUM_CLASSES \
    --save_dir $SAVE_DIR \
    --num_gpus $NUM_GPUS \
    --config $CONFIG_FILE
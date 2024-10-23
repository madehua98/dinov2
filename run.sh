#! /bin/sh
# Set the master address and port
export MASTER_ADDR=localhost  # Replace with the master node's IP address if needed
export MASTER_PORT=29501      # Choose a port that is not in use


# ROOT=("/mnt/madehua/fooddata/vegfru-dataset/veg200_images" "/mnt/madehua/fooddata/vegfru-dataset/fru92_images" "/mnt/madehua/fooddata/Food2k_complete" "/mnt/madehua/fooddata/food-101" "/mnt/madehua/fooddata/VireoFood172/ready_chinese_food" "/mnt/madehua/fooddata/FoodX-251/images")
ROOT=six_dataset_full.txt

export MASTER_PORT=30000
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,4,5,6,7,9 OMP_NUM_THREADS=8 torchrun --nnodes=1 --master_port=$MASTER_PORT --nproc_per_node=6 dinov2/train/train.py \
    --config-file dinov2/configs/train/vitl14.yaml train.dataset_path=RecursiveImageDataset:root=$ROOT train.OFFICIAL_EPOCH_LENGTH=16000 train.batch_size_per_gpu=16 train.output_dir=/media/fast_data/model/checkpoints/food_vitl14/six_full_dataset

## dinov2/configs/train/vitl14.yaml
## dinov2/configs/ssl_food_vitl14.yaml  

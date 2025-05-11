#!/bin/bash
#SBATCH --job-name=cifar10_train
#SBATCH --output=logs/cifar10_%j.out
#SBATCH --error=logs/cifar10_%j.err
# SBATCH --gres=gpu:4
# SBATCH --cpus-per-task=8
# SBATCH --mem=16G
# SBATCH --time=9:00:00


#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=59:00


# Load your modules or activate env
source ~/.bashrc
conda activate dxmi  # or your specific environment

# Change to project directory
cd /home/cz5047/imagegen/Diffusion-by-MaxEntIRL  # replace with actual path

MASTER_PORT=$((10000 + RANDOM % 50000))

# Run training
srun torchrun --nproc_per_node=4 --master-port=$MASTER_PORT train_cifar10.py -- \
  --config configs/cifar10/T10-v2.yaml \
  --dataset configs/cifar10/cifar10.yaml \
  --run 5_9_15_41_t15

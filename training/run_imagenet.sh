#!/bin/bash

#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --mem=150g
#SBATCH -c 8
#SBATCH --partition=rtx6000
#SBATCH --job-name=train-imagenet-rtx6000-imagenet18
#SBATCH --output=train-imagenet-rtx6000-imagenet18.out

# source /ssd003/home/ady/.bashrc
# conda activate /ssd003/home/ady/envnew
source /h/ady/.envnew

ulimit -n 4096
python -m torch.distributed.launch \
--nproc_per_node=4 --nnodes=1 --node_rank=0 \
train_imagenet_nv.py \
--workers=4 --fp16 --logdir /ssd003/home/ady/imagenet_fast --distributed --init-bn0 --no-bn-wd \
--phases "[{'ep': 0, 'sz': 128, 'bs': 256}, {'ep': (0, 8), 'lr': (0.5, 1.0)}, {'ep': (8, 15), 'lr': (1.0, 0.125)}, {'ep': 15, 'sz': 224, 'bs': 112, 'min_scale': 0.087}, {'ep': (15, 25), 'lr': (0.22, 0.022)}, {'ep': (25, 28), 'lr': (0.022, 0.0022)}, {'ep': 28, 'sz': 288, 'bs': 64, 'min_scale': 0.5, 'rect_val': True}, {'ep': (28, 29), 'lr': (0.00125, 0.000125)}]" --skip-auto-shutdown

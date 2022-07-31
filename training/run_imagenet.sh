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
--phases "[{'ep': 0, 'sz': 128, 'bs': 512}, {'ep': (0, 7), 'lr': (1.0, 2.0)}, {'ep': (7, 13), 'lr': (2.0, 0.25)}, {'ep': 13, 'sz': 224, 'bs': 224, 'min_scale': 0.087}, {'ep': (13, 22), 'lr': (0.4375, 0.043750000000000004)}, {'ep': (22, 25), 'lr': (0.043750000000000004, 0.004375)}, {'ep': 25, 'sz': 288, 'bs': 128, 'min_scale': 0.5, 'rect_val': True}, {'ep': (25, 28), 'lr': (0.0025, 0.00025)}]" --skip-auto-shutdown

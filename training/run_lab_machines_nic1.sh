timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0,1,2,3
ulimit -n 4096
nohup python -m torch.distributed.launch \
  --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  train_imagenet_nv.py \
  --data /home/nicolas/data/imagenet \
  --save_path /datadrive1/adam/imagenet_fast_equus \
  --workers=4 --fp16 --logdir /datadrive1/adam/imagenet_fast_equus --distributed --init-bn0 --no-bn-wd \
  --phases "[{'ep': 0, 'sz': 128, 'bs': 512}, {'ep': (0, 7), 'lr': (1.0, 2.0)}, {'ep': (7, 13), 'lr': (2.0, 0.25)}, {'ep': 13, 'sz': 224, 'bs': 224, 'min_scale': 0.087}, {'ep': (13, 22), 'lr': (0.4375, 0.043750000000000004)}, {'ep': (22, 25), 'lr': (0.043750000000000004, 0.004375)}, {'ep': 25, 'sz': 288, 'bs': 128, 'min_scale': 0.5, 'rect_val': True}, {'ep': (25, 28), 'lr': (0.0025, 0.00025)}]" \
  --skip-auto-shutdown \
  >>train_imagenet_main_${timestamp}.txt 2>&1 &
echo train_imagenet_main_${timestamp}.txt
[2] 25920
(python39) ady@nic3:~/code2/imagenet18/training$ echo train_imagenet_main_${timestamp}.txt
train_imagenet_main_2022-07-31-17-13-17-710460612.txt



python -m torch.distributed.launch --nproc_per_node=4 train.py 2>&1 | tee mlu370_x4_ddp_4_log

#!/usr/bin/env bash

# can be trained on two 24G 3090 GPUs

# chairs

#CHECKPOINT_DIR=checkpoints/chairs-PWTF && \
#mkdir -p ${CHECKPOINT_DIR} && \
#CUDA_VISIBLE_DEVICES=0,1 python main_WAFT.py \
#--checkpoint_dir ${CHECKPOINT_DIR} \
#--batch_size 8 \
#--val_dataset chairs \
#--val_iters 5 \
#--lr 4e-4 \
#--image_size 352 480 \
#--summary_freq 100 \
#--val_freq 10000 \
#--save_ckpt_freq 10000 \
#--save_latest_ckpt_freq 1000 \
#--num_steps 100000 \
#2>&1 | tee ${CHECKPOINT_DIR}/train.log


# things
CHECKPOINT_DIR=checkpoints/things-PWTF && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0,1 python main_WAFT.py \
--stage things \
--resume checkpoints/chairs-PWTF/step_100000.pth \
--no_resume_optimizer \
--checkpoint_dir ${CHECKPOINT_DIR} \
--batch_size 4 \
--val_dataset sintel kitti \
--val_iters 5 \
--lr 1.25e-4 \
--image_size 416 736 \
--freeze_bn \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 10000 \
--save_latest_ckpt_freq 1000 \
--num_steps 200000 \
2>&1 | tee ${CHECKPOINT_DIR}/train.log
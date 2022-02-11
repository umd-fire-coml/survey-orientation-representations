#!/bin/bash
python3 training.py \
    --predict rot-y \
    --converter exp-A \
    --epoch 1 \
    --kitti_dir "/home/siyuan/dataset/kitti" \
    --training_record "/home/siyuan/fire/kitti_orientation/weights/testing"
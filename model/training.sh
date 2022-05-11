#!/bin/bash
python3 training.py \
    --predict rot-y \
    --converter exp-A \
    --epoch 100 \
    --batch_size 25 \
    --kitti_dir "/home/siyuan/dataset/kitti" \
    --training_record "/home/siyuan/fire/kitti_orientation/weights/experimentC" \
    --resume True


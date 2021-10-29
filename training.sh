#!/bin/bash
#python3 training.py --predict rot-y --converter rot-y --batch_size 25 --epoch 100 --kitti_dir "/home/siyuan/dataset/kitti" --training_record "/home/siyuan/fire/KITTI-Orientation/angular_loss" --use_angular_loss True
python training.py --predict rot-y --converter tricosine --batch_size 25 --epoch 100 --kitti_dir "/home/siyuan/dataset/kitti" --training_record "/home/siyuan/fire/KITTI-Orientation/depth_map_weights" --add_depth_map True

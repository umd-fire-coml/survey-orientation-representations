#!/bin/bash
# python3 training.py --predict rot-y --converter tricosine --batch_size 25 --epoch 100 --kitti_dir "/home/siyuan/dataset/kitti" --training_record "/home/siyuan/fire/KITTI-Orientation/pos_enc+depth" --add_depth_map True --add_pos_enc True --resume True
python3 training.py --predict rot-y --converter multibin --batch_size 25 --epoch 100 --kitti_dir "/home/siyuan/dataset/kitti" --training_record "/home/siyuan/fire/KITTI-Orientation/multibin_testing" 

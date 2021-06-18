#! /bin/bash
read -p "Enter Weights Directory: " weights_dir
read -p "Enter Port num: " port_num
echo "tensorboard --logdir $weights --port $port_num"
tensorboard --logdir $weights --port $port_num
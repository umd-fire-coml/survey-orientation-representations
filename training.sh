#!/bin/bash
python training.py multibin --epoch 100 --batch_size 25
python training.py tricosine --epoch 100 --batch_size 25
python training.py alpha --epoch 100 --batch_size 25
python training.py rot_y --epoch 100 --batch_size 25

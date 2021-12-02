# A Survey of Orientation Representations for Accurate Deep Rotation Estimation 
## Raymond H. Tu, Siyuan Peng, Valdimir Leung, Richard Gao, Jerry Lan
This is the official implementation for the paper [A survey of Orientation Representations for Accuracte Deep Rotation](www.google.com)

### Table of Conents
- [Environment Setup](### Envrionment Setup)
- [Training](#Training)

### Envrionment Setup
``` bash
# create conda environment based on yml file
conda env update --file environment.yml
# activate conda environment
conda activate KITTI-Orientation
```
Clone git repo:
``` bash
git clone git@github.com:umd-fire-coml/KITTI-orientation-learning.git
```
### Training
Check training.sh for example training script

#### Training Parameter setup:
Training parameters can be config using cmd arguments
- --predict: Specify prediction target. Options are rot-y, alpha
- --converter:  Specify prediction method. Options are alpha, rot-y, tricosine, multibin, voting-bin, single-bin
- --kitti_dir: path to kitti dataset directory. Its subdirectory should have training/ and testing/ Default path is dataset/
- --training_record: root directory of all training record, parent of weights and logs directory. Default path is training_record
- --resume: Resume from previous training under training_record directory
- --add_pos_enc: Add positional encoding to input
- --add_depth_map: Add depth map information to input

For all the training parameter setup, please using
```
python3 model/training.py -h
```
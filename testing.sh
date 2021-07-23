#!/bin/bash
python .\testing.py tricosine .\weights\best\tricosine\epoch-100-loss-0.0177-val_loss-0.0996.h5
python .\testing.py single_bin .\weights\best\single_bin\epoch-100-loss-0.0156-val_loss-0.1002.h5
python .\testing.py voting_bin .\weights\best\voting_bin\epoch-100-loss-0.0176-val_loss-0.3056.h5
python .\testing.py alpha .\weights\best\alpha\epoch-100-loss-0.0245-val_loss-0.2496.h5
python .\testing.py rot_y .\weights\best\rot_y\epoch-100-loss-0.0383-val_loss-0.3258.h5
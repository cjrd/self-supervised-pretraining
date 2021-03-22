#!/bin/bash
DET_CFG=$1
WEIGHTS=$2

python $(dirname "$0")/train_net.py --config-file $DET_CFG \
    --num-gpus 8 MODEL.WEIGHTS $WEIGHTS

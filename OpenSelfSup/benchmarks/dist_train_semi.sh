#!/usr/bin/env bash

set -e
set -x

CFG=$1 # use cfgs under "configs/benchmarks/semi_classification/imagenet_*percent/"
PRETRAIN=$2
PY_ARGS=${@:3}
GPUS=4 # in the standard setting, GPUS=4
export PORT=$(( ( RANDOM % 60000 )  + 1025 ))

if [ "$CFG" == "" ] || [ "$PRETRAIN" == "" ]; then
    echo "ERROR: Missing arguments."
    exit
fi
pretrain_data=$(echo $PRETRAIN | awk -F '/hpt-pretrain/' '{print $2}' | awk -F '/' '{print $1}')
config_data=$(echo $CFG | awk -F '/hpt-pretrain/' '{print $2}' | awk -F '/' '{print $1}')

if [[ "$pretrain_data" == "$config_data" ]]; then
    # a little messy, TODO(cjrd) make this cleaner
    WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g" | rev | cut -d/ -f 2-  | rev )/$(echo $PRETRAIN | rev | cut -d/ -f 3 | rev)/$(echo $PRETRAIN | rev | cut -d/ -f 2 | rev)-$(echo ${CFG%.*} | rev | cut -d/ -f 1 | rev)"
else
    WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g" | rev | cut -d/ -f 2-  | rev )/$(echo $PRETRAIN | rev | cut -d/ -f 3 | rev)-${pretrain_data}/$(echo $PRETRAIN | rev | cut -d/ -f 2 | rev)-$(echo ${CFG%.*} | rev | cut -d/ -f 1 | rev)"
fi



# train
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py \
    $CFG \
    --pretrained $PRETRAIN \
    --work_dir $WORK_DIR --seed 0 --launcher="pytorch" ${PY_ARGS}

#!/usr/bin/env bash

set -e

CFG=$1 # use cfgs under "configs/benchmarks/linear_classification/"
PRETRAIN=$2
PY_ARGS=${@:3} # --resume_from --deterministic
GPUS=${NGPU:-1} # When changing GPUS, please also change imgs_per_gpu in the config file accordingly to ensure a consistent total batch size
echo "USING $GPUS GPUS"
PORT=${PORT:-29500}
SEED=${SEED:-0}

if [ "$CFG" == "" ] || [ "$PRETRAIN" == "" ]; then
    echo "ERROR: Missing arguments."
    exit
fi


CONFIG_DATASET="$(echo $CFG | rev | cut -d/ -f 3 | rev)"
BACKBONE_DATASET="$(echo $PRETRAIN | rev | cut -d/ -f 4 | rev)"

status_suffix=""
if [[ "${CONFIG_DATASET}" != "${BACKBONE_DATASET}" ]]; then
  status_suffix="-${BACKBONE_DATASET}"
fi


# WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"
if [ -z "$WORK_DIR" ]; then
    WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g" | rev | cut -d/ -f 2-  | rev )/$(echo $PRETRAIN | rev | cut -d/ -f 3 | rev)${status_suffix}/$(echo $PRETRAIN | rev | cut -d/ -f 2 | rev)-$(echo ${CFG%.*} | rev | cut -d/ -f 1 | rev)"
fi
echo "WORK_DIR $WORK_DIR"

# train
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py \
    $CFG \
    --pretrained $PRETRAIN \
    --work_dir $WORK_DIR --seed $SEED --launcher="pytorch" ${PY_ARGS}

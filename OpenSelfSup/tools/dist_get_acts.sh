#!/bin/bash

set -x

CFG=$1
CHECKPOINT=$2
PORT=${PORT:-29500}
PY_ARGS=${@:3} # --grab_conv, etc.

WORK_DIR="$(dirname $CHECKPOINT)/"

# test
python tools/get_acts.py \
    $CFG \
    $CHECKPOINT \
    --work_dir $WORK_DIR $PY_ARGS

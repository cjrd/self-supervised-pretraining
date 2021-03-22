#!/bin/bash

export PORT=$(( ( RANDOM % 60000 )  + 1025 ))

while read BACKBONE
do
    echo STARTING $BACKBONE
    benchmarks/dist_train_linear.sh configs/benchmarks/linear_classification/resisc45/r50_last.py ${BACKBONE}
    
done < "${1:-/dev/stdin}"

# RESISC

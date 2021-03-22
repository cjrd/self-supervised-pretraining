#!/bin/bash

array=(bdd chexpert resisc)

for i in "${array[@]}"; do
  python plot_pct_pretrain_modified.py --dataset $i
done

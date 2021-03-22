#!/bin/bash

array=(bdd chest_xray_kids domain_net_quickdraw chexpert resisc)

for i in "${array[@]}"; do
  python plot_basetrain_robust_modified.py --dataset $i
done

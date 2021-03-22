#!/bin/bash

array=(bdd chexpert resisc)

for i in "${array[@]}"; do
  python plot_augmentation.py --dataset $i
done

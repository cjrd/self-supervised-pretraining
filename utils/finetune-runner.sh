#!/usr/bin/env bash

set -e

# INPUT: set of weights to use for the finetuning
# finetune base directory

# for each weight:
# for each config in finetune directory
# do finetune config for the given weight

## this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


## flags and positional arguments
config_dir=""
backbone_weights_dir=""
progress_dir=""
print_usage() {
  printf "Usage: ./finetune-runner.sh -d configs/hpt-pretrain/my-data-shortname/finetune -b directory/with/sub/dirs/that/have/final_backbone.pth  [-t progress/tracking/dir] "
}

while getopts 'd:b:t:' flag; do
  case "${flag}" in
    b) backbone_weights_dir="$PWD/${OPTARG}" ;;
    d) config_dir="$PWD/${OPTARG%/}" ;;
    t) progress_dir="$PWD/${OPTARG%/}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

if [[ "$config_dir" == "" ]]; then
    print_usage
    exit 1
fi

if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo "Must set env var 'CUDA_VISIBLE_DEVICES=X' where X is a single digit"
  exit 1
fi

if [[ "$CUDA_VISIBLE_DEVICES" =~ , ]]; then
  echo "Must set env var 'CUDA_VISIBLE_DEVICES=X' where X is a single digit"
  exit 1
fi


# if [[ ! -f "$DIR/../OpenSelfSup/data/basetrain_chkpts/random_r50.pth" ]]; then
#   wget https://people.eecs.berkeley.edu/~xyyue/random_r50.pth
#   mv random_r50.pth "$DIR/../OpenSelfSup/data/basetrain_chkpts/random_r50.pth"
# else 
#   echo "random_50.pth exists!"
# fi

# echo ${backbone_weights_dir}

# ls ${backbone_weights_dir} | while read -r subdir; do

#   zero_iter_dir=${backbone_weights_dir}/$subdir/0-iters

#   if [[ $subdir = no* ]] ; then
#     mkdir -p $zero_iter_dir
#     ln -fs $PWD/OpenSelfSup/data/basetrain_chkpts/random_r50.pth $zero_iter_dir/final_backbone.pth
#   elif [[ $subdir = moco* ]]; then
#     mkdir -p $zero_iter_dir
#     ln -fs $PWD/OpenSelfSup/data/basetrain_chkpts/moco_v2_800ep.pth $zero_iter_dir/final_backbone.pth

#   elif [[ $subdir = imagenet* ]]; then
#     mkdir -p $zero_iter_dir
#     ln -fs $PWD/OpenSelfSup/data/basetrain_chkpts/imagenet_r50_supervised.pth $zero_iter_dir/final_backbone.pth
#   fi

# done
# echo $dirs

# set some basic env vars
## random port so we (probably) don't collide
export PORT=$(( ( RANDOM % 60000 )  + 1025 ))
# navigate to the OpenSelfSup, since relative paths assume this (I think...)
cd "$DIR/../OpenSelfSup" || exit 1

if [[ "$progress_dir" == "" ]]; then
  progress_dir="${PWD}/.finetune-progress"
fi
mkdir -p "$progress_dir"

# Loop through all of the configs
find_str="*-finetune.py"
find "$config_dir" -name ${find_str} | while read -r CFG; do

  # loop through all backbones for each config
  # special_final_bkbn.pth
  find "${backbone_weights_dir}" -name "special_final_bkbn.pth" | while read -r WEIGHTS; do
    progress_name=$(echo ${CFG%.py} | awk -F "OpenSelfSup/" '{print $NF}' | sed "s/\//-/g")
    progress_name=$(echo ${WEIGHTS%final_backbone.pth} | awk -F "OpenSelfSup/" '{print $NF}' | sed "s/\//-/g")"${progress_name}"

    procfile=${progress_dir}/${progress_name}.proc
    donefile=${progress_dir}/${progress_name}.done
    
    if [[ -f "$procfile" || -f "$donefile" ]]; then
      echo "${donefile%.done} exists, skipping"
      continue
    fi  

    touch "$procfile"

    set +e
    NGPU=1 benchmarks/train_linear.sh "$CFG" "$WEIGHTS"

    ret_val=$?
    if [ $ret_val -ne 0 ]; then
        echo "Bad exit status from benchmarks/dist_train_semi.sh $CFG $WEIGHTS"
        rm "$procfile"        
        exit 1
    fi


    set -e

    rm "$procfile"
    touch "$donefile"

  done  
done

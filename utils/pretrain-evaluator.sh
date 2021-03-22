#!/usr/bin/env bash

set -e

# Future goals for this script:
# * Pass in a results dir and a config and it will do everything, meaning:
# * Be able to do all types of evaluation of a representation (e.g. linear, semi-sup, etc)
# * be able to do a full grid search
# * be able to select the top results and run it 3 times

## flags and positional arguments
backbone_dir=""
config_dir=""
eval_all_samples=""
skip_base_eval=""

print_usage() {
  printf "Usage: ./pretrain-evaluator.sh [-a (eval all samples, else will only do s0 sample config)] [-q (skip base eval)] -b work_dirs/hpt-pretrain/my-data-shortname -d configs/hpt-pretrain/my-data-shortname"
}

while getopts 'd:b:aq' flag; do
  case "${flag}" in
    a) eval_all_samples="true" ;;
    b) backbone_dir="$PWD/${OPTARG%/}" ;;
    d) config_dir="$PWD/${OPTARG%/}" ;;
    q) skip_base_eval="true" ;;
    *) print_usage
       exit 1 ;;
  esac
done


if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo "Must set env var 'CUDA_VISIBLE_DEVICES=X' where X is a single digit"
  exit 1
fi

if [[ "$CUDA_VISIBLE_DEVICES" =~ , ]]; then
  echo "Must set env var 'CUDA_VISIBLE_DEVICES=X' where X is a single digit"
  exit 1
fi

## this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ ! -f "$DIR/../OpenSelfSup/data/basetrain_chkpts/random_r50.pth" ]]; then
  wget https://people.eecs.berkeley.edu/~xyyue/random_r50.pth
  mv random_r50.pth "$DIR/../OpenSelfSup/data/basetrain_chkpts/random_r50.pth"
else 
  echo "random_50.pth exists - will not redownload"
fi


CONFIG_DATASET="$(echo $config_dir | rev | cut -d/ -f 1 | rev)"
BACKBONE_DATASET="$(echo $backbone_dir | rev | cut -d/ -f 1 | rev)"

echo "CONFIG:" ${CONFIG_DATASET}
echo "BACKBONE_DATASET: " ${BACKBONE_DATASET}

status_suffix=""
if [[ "${CONFIG_DATASET}" != "${BACKBONE_DATASET}" ]]; then
  status_suffix="-${CONFIG_DATASET}"

  # Here we assume that the base eval on the other dataset is already done.
  skip_base_eval="true"
fi

if [[ "$config_dir" == "" || "$backbone_dir" == "" ]]; then
    print_usage
    exit 1
fi

# check it's the right dir
baseconfig="${config_dir}/base*config.py"
if [ ! -f $baseconfig ]; then
    echo "The provided config dir does not exist: try passing in the full filepath"
    echo "does not exist: $config_dir/base*config.py"
    exit 1
fi

# TODO add check for backbone dir

# set some basic env vars
## random port so we (probably) don't collide
export PORT=$(( ( RANDOM % 60000 )  + 1025 ))
# navigate to the OpenSelfSup, since relative paths assume this (I think...)
cd "$DIR/../OpenSelfSup" || exit 1


# Loop through all of the learning rate configs
find_str="*-s0.py"
if [[ "$eval_all_samples" == "true" ]]; then
  find_str="*-s[0-9].py"
fi
backbone_str="${BBONESTR:-final_backbone.pth}"

# go through the config directory
find "$backbone_dir" -name "*-iters" -type d | while read itdir; do
  iters=$(basename "$itdir" | awk -F "-" '{print $1}')
  iters=${iters%bn}
  iters=${iters#*lr}
  lastiter="$itdir/iter_${iters}.pth"
  if [[ -f $lastiter && ! -f "$itdir/final_backbone.pth" ]]; then
    echo "$itdir/final_backbone.pth does not exist. extracting now"
    python tools/extract_backbone_weights.py $lastiter "$itdir/final_backbone.pth"
  fi
done
#
# Loop through all of the linear eval configs
#
find "$config_dir"  -type f -name "${find_str}" | while read fname; do
    usename=$(echo "$fname" | awk -F 'OpenSelfSup/' '{print $NF}')
    confname=$(basename "$usename")
    confname="${confname%.py}${status_suffix}"
    # which sample is this?
    sample=$(echo "${fname%.py}" | awk -F '-s' '{print $NF}')
    echo evaluating sample $sample

  #
  # do the base weight evaluations
  #
    if [[ "$skip_base_eval" == "" ]]; then
      echo "checking base eval"
      
      # do the base weights
      echo 'Scanning base weights (add -b flag to not include base weight evaluations)'
      find "$backbone_dir" -name "${backbone_str}" -type f | while read bname; do
        # extract the base weight from the filepath (3rd to last dir entry) #brittle
        bw=$(echo "$bname" | awk -F '/' '{print $(NF-2)}' | sed 's/_basetrain$//')
	
        # have to set a custom work dir
        fname_end=$(basename ${fname%.py})
        wdir=$(dirname $fname | sed 's/configs/work_dirs/')
        bdir=$(dirname "$bname" | rev | cut -d/ -f 2- | rev)
	  
        if [[ "$bw" == "no" ]]; then
	    continue
	    # add 0-iter if this is the no basetrain
	    zero_iter_dir=${bdir}/0-iters
	    mkdir -p $zero_iter_dir
	    weights="$DIR/../OpenSelfSup/data/basetrain_chkpts/random_r50.pth"
	    ln -fs $DIR/../OpenSelfSup/data/basetrain_chkpts/random_r50.pth $zero_iter_dir/final_backbone.pth
	else
	    weights="$DIR/../OpenSelfSup/data/basetrain_chkpts/$bw.pth"
        fi
        wdir="${wdir}/${bw}_basetrain/0-iters-${fname_end}"
	mkdir -p $wdir
	
        donefile=${wdir}/iter_5000.pth
        procfile="${wdir}.proc"
        if [[ -f "$procfile" ]]; then
            echo "${procfile} exists, skipping"
            continue
        fi

        if [[ -f "$donefile" ]]; then
            echo "${donefile} exists, skipping"
            continue
        fi


        touch $procfile
        echo "processing base $weights"

        set +e
	set -x
	if [[ "${fname}" =~ chexpert|coco|pascal|viper ]]; then
	    echo "Multi-label evaluation - not doing distributed training"
	    traincmd=benchmarks/train_linear.sh
	else
	    echo "Single label evaluation"
	    traincmd=benchmarks/dist_train_linear.sh
	fi
	set +x
        NGPU=1 WORK_DIR=${wdir} SEED=${sample} $traincmd "${fname}" "${weights}"
        ret_val=$?

        if [ $ret_val -ne 0 ]; then
            echo "Bad exit status from: WORK_DIR=${wdir} SEED=${sample} benchmarks/dist_train_linear.sh ${fname} ${weights}"
            rm -f "$procfile"
            exit 1
        fi
        set -e

        rm -f "$procfile"
      done
    fi

    #
    # loop through all of the final backbones
    #
    find "$backbone_dir" -name "${backbone_str}" -type f | while read bname; do
        bdir=$(dirname "$bname")

        CFG=${fname}
        PRETRAIN=${bname}
        wdir="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g" | rev | cut -d/ -f 2-  | rev )/$(echo $PRETRAIN | rev | cut -d/ -f 3 | rev)${status_suffix}/$(echo $PRETRAIN | rev | cut -d/ -f 2 | rev)-$(echo ${CFG%.*} | rev | cut -d/ -f 1 | rev)"

	mkdir -p $wdir

        donefile=${wdir}/iter_5000.pth
        procfile="${wdir}.proc"

        if [[ -f "$donefile" ]]; then
            echo "${donefile} exists, skipping"
            continue
        fi
        if [[ -f "$procfile" ]]; then
            echo "${procfile} exists, skipping"
            continue
        fi

        touch "$procfile"
        echo "processing $bname"

        set +e
	
	if [[ "${fname}" =~ chexpert|coco|pascal|viper ]]; then
	    echo "Multi-label evaluation - not doing distributed training"
	    traincmd=benchmarks/train_linear.sh
	else
	    echo "Single label evaluation"
	    traincmd=benchmarks/dist_train_linear.sh
	fi
	set +x
	
        NGPU=1 SEED=${sample} $traincmd "${fname}" "${bname}"

        ret_val=$?
        if [ $ret_val -ne 0 ]; then
            echo "Bad exit status from: SEED=${sample} benchmarks/dist_train_linear.sh ${fname} ${bname}"
            rm -f "$procfile"
            exit 1
        fi
        set -e

        rm -f "$procfile"
    done
done

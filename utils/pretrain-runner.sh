#!/bin/bash

set -e

## flags and positional arguments
trial="false"
config_dir=""
byol="false"
print_usage() {
  printf "Usage: ./pretrain-runner.sh [-t trial] [-n batchnorm only] -d configs/hpt-pretrain/my-data-shortname"  
}

while getopts 'd:tbn' flag; do
  case "${flag}" in
      t) trial='true' ;;
      n) bn='true' ;;
      b) byol='true' ;;      
      d) config_dir="$PWD/${OPTARG}" ;;
      *) print_usage
       exit 1 ;;
  esac
done

if [[ "$config_dir" == "" ]]; then
    echo "must supply config"
    print_usage
    exit 1
fi

# check it's the right dir
baseconfig=${config_dir}/base*config.py
if [ ! -f $baseconfig ]; then
    echo "The provided config dir does not exist: try passing in the full filepath"
    echo "does not exist: $config_dir/base*config.py"
    exit 1
fi


# set some basic env vars
## random port so we (probably) don't collide
export PORT=$(( ( RANDOM % 60000 )  + 1025 ))
## this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# navigate to the OpenSelfSup, since relative paths assume this (I think...)
cd "$DIR/../OpenSelfSup" || exit 1

NGPU=${NGPU:-4}

# Loop through all of the configs
find_str="*-iters.py"
if [[ "$trial" == "true" ]]; then
    find_str="50-iters.py"
fi
if [[ "$bn" == "true" ]]; then
    find_str="*bn-iters.py"
fi

find "$config_dir" -name ${find_str} | sort -R | while read fname; do
    if [[ "$byol" == "true" ]]; then
        if [[ "$fname" == *"byol"* ]]; then
            echo $fname
        else
            continue
        fi
    fi
    usename=$(echo "$fname" | awk -F 'OpenSelfSup/' '{print $NF}')
    confname=$(basename "$usename")
    confname="${confname%.py}"

    statusdir=$(dirname "$usename")
    statusdir="${statusdir}/.pretrain-status"
    mkdir -p "${statusdir}"

    status_name="${statusdir}/${confname}"
    procfile="${status_name}.proc"
    donefile="${status_name}.done"
    if [[ -f "$procfile" || -f "$donefile" ]]; then
        echo "${status_name} exists, skipping"
        continue
    fi
    touch "$procfile"
    echo "processing $fname"

    # don't exit on error so we can catch error
    set +e
    tools/dist_train.sh "$fname" ${NGPU}
    ret_val=$?
    if [ $ret_val -ne 0 ]; then
        echo "Bad exit status from tools/dist_train.sh $fname"
        rm "$procfile"        
        exit 1
    fi
    touch "$donefile"
    rm "$procfile"

    result_dir=$(echo $fname | awk -F 'configs/' '{print $NF}')
    result_dir="work_dirs/${result_dir%.py}"
    iter_num=$(echo $fname | awk -F '/' '{ print $NF }' | awk -F '-' '{print $1}'| sed 's/[^0-9]*//g')
    final_weight="$result_dir/iter_${iter_num}.pth"
    final_backbone="$result_dir/final_backbone.pth"
    echo "extracting final weights from: $final_weight"
    echo "extracting final weights into: $final_backbone"

    python tools/extract_backbone_weights.py $final_weight $final_backbone

    set -e
    if [[ "$trial" == "true" ]]; then
        printf "\n\nTrial flag provided, exiting after 1 pretraining\n\n"
        break
    fi
done


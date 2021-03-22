#!/bin/bash
# here it is
# QUICKSTART:
# > cp templates/pretraining-config-template.sh pretrain-configs/my-dataset-config.sh
# > # edit pretrain-configs/my-dataset-config.sh
# > ./gen-pretrain-project.sh pretrain-configs/my-dataset-config.sh
#
# See detailed instructions in the README ('Pretraining with a New Dataset')
#
# OUTPUT:
# This script creates the following config structure
# configs/hpt-pretrain/${shortname}/base-${shortname}-config.py
# configs/hpt-pretrain/${shortname}/{basetrain_weights_name}_basetrain/{configs-names}.py
# configs/hpt-pretrain/${shortname}/no_basetrain/{configs-names}.py
# configs/hpt-pretrain/${shortname}/supervised_basetrain/{configs-names}.py

set -e

if [[ $# != 1 ]]; then
    echo
    echo "You must supply the pretraining config input (see README.md: 'Pretraining with a New Dataset')"
    echo "./gen-pretrain-project.sh pretrain-configs/myconfig.sh"
    exit 1
fi
# shellcheck source="templates/pretraining-config-template.sh"
# shellcheck disable=SC1091
. "$1"

# setup the config dir
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
openselfsup_dir="${DIR}/../OpenSelfSup"
configdir_base="${openselfsup_dir}/configs/hpt-pretrain/"
mkdir -p ${configdir_base}
configdir_base=$(cd "${configdir_base}"; pwd)

configdir="${configdir_base}/${shortname}"
if [[ -d "${configdir}" && -z "${HPT_OVERWRITE}" ]]; then
    echo
    echo "ERROR"
    echo "DIRECTORY EXISTS: ${configdir}"
    read -r -p "This will overwrite existing configs but will not delete anything else. Proceed? [Y/n]" resp
    resp=$(echo -e "${resp}" | tr '[:upper:]' '[:lower:]')
    if [ "$resp" = "n" ]; then
        echo "Bailing"
        exit 1
    fi
fi

# confirm all the settings
# carry out all of the actions at the end: nice system etiquette
echo
echo "will create config dir: ${configdir}"
echo "using pixel means: ${pixel_means}"
echo "using pixel stds: ${pixel_stds}"
echo "using train+val data in file: ${train_val_combined_list_path}"
echo "using train data in file: ${train_list_path}"
echo "using val data in file: ${val_list_path}"
echo "using test data in file: ${test_list_path}"
echo "with base data path: ${base_data_path}"
echo "using basetrain weights: ${basetrain_weights[@]}"
echo "with crop size: $crop_size"
echo

read -r -p "Proceed? [Y/n]" resp
resp=$(echo -e "${resp}" | tr '[:upper:]' '[:lower:]')
if [ "$resp" = "n" ]; then
    echo "Bailing"
    exit 1
fi

# Carry out the actions

# create config dirs

# bash template file
# from https://stackoverflow.com/questions/2914220/bash-templating-how-to-build-configuration-files-from-templates-with-bash
output=$(perl -p -e 's/\$\{([^}]+)\}/defined $ENV{$1} ? $ENV{$1} : $&/eg' < "${DIR}/templates/config-template.py")
mkdir -p ${configdir}
echo "$output" > "${configdir}/base-${shortname}-config.py"

for btw in "${basetrain_weights[@]}"; do
    if [[ "$btw" != "" ]]; then
        tmp=$(basename $btw) 
        basetrain_name="${tmp%.*}"
    else
        basetrain_name="no"
    fi

    model_name=$(basename "$btw")
    if [[ "$model_name" == "final_backbone.pth" ]]; then
        model_dir=$(dirname "$btw")
        newname=$(echo "$model_dir" | awk -F '/' '{ split($NF, ss, "-"); printf "%s-%s_%sit.pth", $(NF-1), $(NF-2), ss[1] }')
        # "work_dirs/hpt-pretrain/resisc/moco_v2_800ep_basetrain/50000-iters/final_backbone.pth"
        # "work_dirs/hpt-pretrain/resisc/moco_v2_800ep_basetrain/50000-iters/moco_v2_800ep_resisc_50000it.pth"
        echo "Renaming final_backbone.pth to ${newname}"
        basetrain_name=${newname%.*}
        pretrained=$newname
        btw=$model_dir/$newname
        cd ${openselfsup_dir}/${model_dir}; ln -sfn final_backbone.pth ${newname}; cd -
        export pretrained
    fi

    basetrain_config_dir="$configdir/${basetrain_name}_basetrain"
    mkdir -p "$basetrain_config_dir"
    echo "creating config dir for: ${basetrain_name} basetrain"

    # now generate all of the different iter trainings for basetrain
    iters_config_dirs="${basetrain_config_dir}"
    export pretrained="${btw}"
    for ep_config_dir in $iters_config_dirs; do
        mkdir -p "$ep_config_dir"

        # no basetraining
        if [[ "$pretrained" == "" ]]; then  
            iters=$no_bt_iters
        else
            iters=$bt_iters
        fi
        
        echo "making configs for ${ep_config_dir} iters: $iters"

        for iter in $(echo $iters | sed "s/,/ /g"); do
            echo $iter
            export iter
            output=$(perl -p -e 's/\$\{([^}]+)\}/defined $ENV{$1} ? $ENV{$1} : $&/eg' < "${DIR}/templates/iter-config-template.py")
	        # remove the pretrained model if training from scratch
	        if [[ "$pretrained" == "" ]]; then
    		    output=$(echo "$output" | awk '!/pretrained/')
                # checkpoint the no bt more often
                output=$(printf '%s\ncheckpoint_config = dict(interval=total_iters//2)' "$output")

	        fi

            echo "$output" > "${ep_config_dir}/${iter}-iters.py"
        done
    done
done


# commented out  eval section means we don't need to gen these
if [[ -z "${num_classes}" ]]; then
    echo "Evaluation section is missing -- will now exit"
    exit 0
fi

# now generate all of the linear eval configs
# starting with the base config
linear_eval_dir="$configdir/linear-eval"
mkdir -p "$linear_eval_dir"

# set optional variables to defaults
image_head_class_type=${image_head_class_type:-"ImageNet"}
export image_head_class_type
bce_string=${bce_string:-""}
export bce_string
class_map=${class_map:-""}
export class_map
dataset_type=${dataset_type:-"ClassificationDataset"}
export dataset_type
eval_params=${eval_params:-"dict(topk=(1,5))"}
export eval_params

output=$(perl -p -e 's/\$\{([^}]+)\}/defined $ENV{$1} ? $ENV{$1} : $&/eg' < "${DIR}/templates/linear-eval-config-template.py")
echo "$output" > "${linear_eval_dir}/linear-eval-base.py"
export learn_rate=${linear_learn_rate}
output=$(perl -p -e 's/\$\{([^}]+)\}/defined $ENV{$1} ? $ENV{$1} : $&/eg' < "${DIR}/templates/linear-eval-lr-template.py")
for ((i=0; i<linear_reruns; i++))
do
    echo "$output" > "${linear_eval_dir}/linear-eval-lr${learn_rate}-s${i}.py"
done


# now generate the finetuning configs:
# LR in {0.1, 0.01}
# Schedule in {2500 steps, 90 epochs, etc}
ft_eval_dir="$configdir/finetune"
mkdir -p "$ft_eval_dir"

# fientune directory structure
## finetune/1000-labels
## finetune/100-labels
## etc
for nlabel in $(echo $ft_num_train_labels | sed "s/,/ /g"); do
    nlabel_dir="${ft_eval_dir}/${nlabel}-labels"
    mkdir -p "$nlabel_dir"

    if [[ "${nlabel}" == "all" ]]; then
        export ft_train_list_path="${train_list_path}"
    else
        export ft_train_list_path="${train_list_path%.txt}-${nlabel}.txt"
    fi
    export ft_val_list_path="${val_list_path}"
    export ft_test_list_path="${test_list_path}"
    output=$(perl -p -e 's/\$\{([^}]+)\}/defined $ENV{$1} ? $ENV{$1} : $&/eg' < "${DIR}/templates/finetune-eval-config-template.py")
    echo "$output" > "${nlabel_dir}/finetune-eval-base.py"

    # iterate over schedules and learning rates
    for ft_lr in $(echo $ft_lrs | sed "s/,/ /g"); do

        export ft_lr=$ft_lr
        title_lr=$(echo $ft_lr | sed "s/\./_/g")
        # by iter
        export by_epoch=False
        export by_iter=True
        export lr_warmup_by_epoch=False
        export val_interval=25
        export log_interval=1
        export lr_warmup=$ft_by_iter_lr_warmup
        export lr_steps=$ft_by_iter_lr_steps
        export total_line="total_iters=${ft_by_iter}"
        export total_val="${ft_by_iter}"
        output=$(perl -p -e 's/\$\{([^}]+)\}/defined $ENV{$1} ? $ENV{$1} : $&/eg' < "${DIR}/templates/finetune-sched-lr-config-template.py")
        echo "$output" > "${nlabel_dir}/${ft_by_iter}-iter-${title_lr}-lr-finetune.py"

        # by epoch
        export by_epoch=True
        export by_iter=False
        export lr_warmup_by_epoch=True
        export val_interval=1
        export log_interval=1
        export lr_warmup=$ft_by_epoch_lr_warmup
        export lr_steps=$ft_by_epoch_lr_steps
        export total_line="total_epochs=${ft_by_epoch}"
        export total_val="${ft_by_epoch}"
        output=$(perl -p -e 's/\$\{([^}]+)\}/defined $ENV{$1} ? $ENV{$1} : $&/eg' < "${DIR}/templates/finetune-sched-lr-config-template.py")
        echo "$output" > "${nlabel_dir}/${ft_by_epoch}-epoch-${title_lr}-lr-finetune.py"

    done    
done

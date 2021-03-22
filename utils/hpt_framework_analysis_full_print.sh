#!/bin/bash

# usage: sample usage: update_var clipart sketch 5000 5000
update_var () {

    l1000_no_basetrain="[.[] | select( (.basetrain == \"none\" or .basetrain == \"no\") and .pretrain_iters==\"0\"  and .subset==\"1000-labels\") | .file, .result ]"
    all_no_basetrain="[.[] | select( (.basetrain == \"none\" or .basetrain == \"no\") and .pretrain_iters==\"0\"  and .subset==\"all-labels\") | .file, .result ]" 

    l1000_base_imagenet="[.[] | select(.result_type | contains (\"finetune\")) | select(.file | contains (\"$2\")) | select(.subset | contains(\"1000\")) | select(.basetrain | contains(\"imagenet\")) | select(.pretrain_iters == \"0\") | .file, .result]"
    l1000_base_moco="[.[] |select(.result_type | contains (\"finetune\")) |  select(.file | contains (\"$2\")) | select(.subset | contains(\"1000\")) | select(.basetrain | contains(\"moco\")) | select(.pretrain_iters == \"0\") | .file, .result] "

    l1000_base_source_imagenet="[.[] | select(.result_type | contains (\"finetune\")) | select(.file | contains (\"$1\")) | select(.subset | contains(\"1000\")) | select(.basetrain | contains(\"imagenet\")) | select(.pretrain_iters == \"$3\") | .file, .result]"
    l1000_base_source_moco="[.[] | select(.result_type | contains (\"finetune\")) | select(.file | contains (\"$1\")) | select(.subset | contains(\"1000\")) | select(.basetrain | contains(\"moco\")) | select(.pretrain_iters == \"$3\") | .file, .result] "

    l1000_base_target_imagenet="[.[] | select(.result_type | contains (\"finetune\")) | select(.file | contains (\"$2\")) | select(.subset | contains(\"1000\")) | select(.basetrain | contains(\"imagenet\")) | select(.pretrain_iters == \"$4\") | .file, .result] "
    l1000_base_target_moco="[.[] | select(.result_type | contains (\"finetune\")) | select(.file | contains (\"$2\")) | select(.subset | contains(\"1000\")) | select(.basetrain | contains(\"moco\")) | select(.pretrain_iters == \"$4\") | .file, .result] "



    all_base_imagenet="[.[] | select(.result_type | contains (\"finetune\")) | select(.file | contains (\"$2\")) | select(.subset | contains(\"all\")) | select(.basetrain | contains(\"imagenet\")) | select(.pretrain_iters == \"0\") | .file, .result] "
    all_base_moco="[.[] | select(.result_type | contains (\"finetune\")) | select(.file | contains (\"$2\")) | select(.subset | contains(\"all\")) | select(.basetrain | contains(\"moco\")) | select(.pretrain_iters == \"0\") | .file, .result] "

    all_base_source_imagenet="[.[] |select(.result_type | contains (\"finetune\")) |  select(.file | contains (\"$1\")) | select(.subset | contains(\"all\")) | select(.basetrain | contains(\"imagenet\")) | select(.pretrain_iters == \"$3\") | .file, .result] "
    all_base_source_moco="[.[] | select(.result_type | contains (\"finetune\")) | select(.file | contains (\"$1\")) | select(.subset | contains(\"all\")) | select(.basetrain | contains(\"moco\")) | select(.pretrain_iters == \"$3\") | .file, .result] "

    all_base_target_imagenet="[.[] | select(.result_type | contains (\"finetune\")) | select(.file | contains (\"$2\")) | select(.subset | contains(\"all\")) | select(.basetrain | contains(\"imagenet\")) | select(.pretrain_iters == \"$4\") | .file, .result] "
    all_base_target_moco="[.[] | select(.result_type | contains (\"finetune\")) | select(.file | contains (\"$2\")) | select(.subset | contains(\"all\")) | select(.basetrain | contains(\"moco\")) | select(.pretrain_iters == \"$4\") | .file, .result] "

}

sources=(domain_net_clipart chexpert resisc)
targets=(domain_net_sketch chest_xray_kids ucmerced)

source_bests=(5000 5000 5000)
target_bests=(5000 50000 500)

prefixes=(all l1000)
suffixes=(base_imagenet base_moco base_source_imagenet base_source_moco base_target_imagenet base_target_moco)

#### Double Hierarchy
l1000_base_source_target_moco_ucmerced="[.[] | select(.result_type | contains (\"finetune\")) | select( .file | contains (\"resisc-ucmerced\"))  | select(.file | contains(\"finetune\"))  | select(.file | contains(\"moco\")) | select(.file | contains(\"/1000-labels\")) | .file, .result] "
all_base_source_target_moco_ucmerced="[.[] | select(.result_type | contains (\"finetune\")) | select( .file | contains (\"resisc-ucmerced\"))  | select(.file | contains(\"finetune\"))  | select(.file | contains(\"moco\")) | select(.file | contains(\"/all-labels\")) | .file, .result] "

l1000_base_source_target_moco_chest_xray_kids="[.[] | select(.result_type | contains (\"finetune\")) | select( .file | contains (\"-chexpert-\"))  | select(.file | contains(\"finetune\"))  | select(.file | contains(\"moco\")) | select(.file | contains(\"1000-labels\")) | .file, .result] "
all_base_source_target_moco_chest_xray_kids="[.[] | select(.result_type | contains (\"finetune\")) | select( .file | contains (\"-chexpert-\"))  | select(.file | contains(\"finetune\"))  | select(.file | contains(\"moco\")) | select(.file | contains(\"all-labels\")) | .file, .result] "

l1000_base_source_target_moco_domain_net_sketch="[.[] | select(.result_type | contains (\"finetune\")) | select( .file | contains (\"_clipart-\"))  | select(.file | contains(\"finetune\"))  | select(.file | contains(\"moco\")) | select(.file | contains(\"1000-labels\"))  | .file, .result] "
all_base_source_target_moco_domain_net_sketch="[.[] |select(.result_type | contains (\"finetune\")) |  select( .file | contains (\"_clipart-\"))  | select(.file | contains(\"finetune\"))  | select(.file | contains(\"moco\")) | select(.file | contains(\"all-labels\")) | .file, .result] "

l1000_base_source_target_imagenet_ucmerced="[.[] | select(.result_type | contains (\"finetune\")) | select( .file | contains (\"-all-labels\"))  | select(.file | contains(\"finetune\"))  | select(.file | contains(\"imagenet\")) | select(.file | contains(\"/1000-labels\"))  | .file, .result] "
all_base_source_target_imagenet_ucmerced="[.[] | select(.result_type | contains (\"finetune\")) | select( .file | contains (\"-all-labels\"))  | select(.file | contains(\"finetune\"))  | select(.file | contains(\"imagenet\")) | select(.file | contains(\"/all-labels\"))  | .file, .result] "

l1000_base_source_target_imagenet_chest_xray_kids=${l1000_base_source_target_imagenet_ucmerced}
all_base_source_target_imagenet_chest_xray_kids=${all_base_source_target_imagenet_ucmerced}

l1000_base_source_target_imagenet_domain_net_sketch=${l1000_base_source_target_imagenet_ucmerced}
all_base_source_target_imagenet_domain_net_sketch=${all_base_source_target_imagenet_ucmerced}

for i in "${!sources[@]}"; do 

    printf "========================================== %s %s %s %s\n" "${sources[$i]}" "${targets[$i]}" "${source_bests[$i]}" "${target_bests[$i]}" 

    result_file="./results/${targets[$i]}_results.json"
    update_var "${sources[$i]}" "${targets[$i]}" "${source_bests[$i]}" "${target_bests[$i]}" 

    for prefix in ${prefixes[@]}; do
        varname="${prefix}_no_basetrain"
        test_top1=$(jq "${!varname}" "$result_file")
        echo $varname
        echo "${!varname}"
        echo -e $test_top1 
        echo 

        for suffix in ${suffixes[@]}; do
            varname="${prefix}_${suffix}"
            command=${!varname}
            
            test_top1=$(jq "${!varname}" "$result_file")
            echo $varname
            echo "${!varname}"
            echo -e $test_top1 
            echo 
        done
        varname="${prefix}_base_source_target_imagenet_${targets[$i]}"
        test_top1=$(jq "${!varname}" "$result_file")
        echo  "${prefix}_base_source_target_imagenet"
        echo "${!varname}"
        echo -e $test_top1
        echo 

        varname="${prefix}_base_source_target_moco_${targets[$i]}"
        test_top1=$(jq "${!varname}" "$result_file")
        # echo $command
        echo "${prefix}_base_source_target_moco"
        echo "${!varname}"
        echo -e $test_top1 
        echo 
    done

done
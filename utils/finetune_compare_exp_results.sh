#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
datas=(bdd chest_xray_kids chexpert coco_2014 domain_net_clipart domain_net_infograph domain_net_painting domain_net_real domain_net_sketch domain_net_quickdraw resisc  ucmerced viper flowers pascal)

# all_no_basetrain="[.[] | select( (.basetrain == \"none\" or .basetrain == \"no\") and .pretrain_iters==\"0\"  and  (.subset | contains(\"all\"))) | select(.result)] | max_by(.result | tonumber) | .result"
# all_base_moco="[.[] | select( (.basetrain == \"moco_v2_800ep\") and .pretrain_iters==\"0\"  and (.subset | contains(\"all\"))) | select(.result)] | max_by(.result | tonumber) | .result"
# all_base_sup="[.[] | select( (.basetrain == \"imagenet_r50_supervised\") and .pretrain_iters==\"0\"  and (.subset | contains(\"all\"))) | select(.result)]  | max_by(.result | tonumber) | .result"

# all_top_none_moco="[.[] | select((.basetrain == \"none\" or .basetrain ==\"no\") and .pretrain_iters!=\"0\" and (.subset | contains(\"all\"))) | select(.result)] | max_by(.result | tonumber) | .result"
# all_top_moco_moco="[.[] | select((.basetrain == \"moco_v2_800ep\") and .pretrain_iters!=\"0\" and (.subset | contains(\"all\"))) | select(.result)] | max_by(.result | tonumber) | .result"
# all_top_sup_moco="[.[] | select((.basetrain == \"imagenet_r50_supervised\") and .pretrain_iters!=\"0\" and (.subset | contains(\"all\"))) | select(.result)] | max_by(.result | tonumber) | .result"
# all_bn="[.[] | select(.result_type | contains (\"finetune\")) | select( .file | contains (\"bn\")) | select(.subset | contains(\"all\")) | select(.result)] | max_by(.result | tonumber) | .result"


# l1000_no_basetrain="[.[] | select( (.basetrain == \"none\" or .basetrain == \"no\") and .pretrain_iters==\"0\"  and (.subset | contains(\"1000\"))) | select(.result)] | max_by(.result | tonumber) | .result"
# l1000_base_moco="[.[] | select( (.basetrain == \"moco_v2_800ep\") and .pretrain_iters==\"0\"  and (.subset | contains(\"1000\"))) | select(.result)] | max_by(.result | tonumber) | .result"
# l1000_base_sup="[.[] | select( (.basetrain == \"imagenet_r50_supervised\") and .pretrain_iters==\"0\"  and (.subset | contains(\"1000\"))) | select(.result)]  | max_by(.result | tonumber) | .result"

# l1000_top_none_moco="[.[] | select((.basetrain == \"none\" or .basetrain ==\"no\") and .pretrain_iters!=\"0\" and (.subset | contains(\"1000\"))) | select(.result)] | max_by(.result | tonumber) | .result"
# l1000_top_moco_moco="[.[] | select((.basetrain == \"moco_v2_800ep\") and .pretrain_iters!=\"0\" and (.subset | contains(\"1000\"))) | select(.result)] | max_by(.result | tonumber) | .result"
# l1000_top_sup_moco="[.[] | select((.basetrain == \"imagenet_r50_supervised\") and .pretrain_iters!=\"0\" and (.subset | contains(\"1000\"))) | select(.result)] | max_by(.result | tonumber) | .result"
# l1000_bn="[.[] | select(.result_type | contains (\"finetune\")) | select( .file | contains (\"bn\")) | select(.subset | contains(\"1000\")) | select(.result)] | max_by(.result | tonumber) | .result"

all_no_basetrain="[.[] | select( (.basetrain == \"none\" or .basetrain == \"no\") and .pretrain_iters==\"0\"  and  (.subset | contains(\"all\"))) | select(.result) | select(.result_type | contains(\"finetune\"))] | max_by(.result | tonumber) | .result"
all_base_moco="[.[] | select( (.basetrain == \"moco_v2_800ep\") and .pretrain_iters==\"0\"  and (.subset | contains(\"all\"))) | select(.result)] | max_by(.result | tonumber) | .result"
all_base_sup="[.[] | select( (.basetrain == \"imagenet_r50_supervised\") and .pretrain_iters==\"0\"  and (.subset | contains(\"all\"))) | select(.result)]  | max_by(.result | tonumber) | .result"

all_top_none_moco="[.[] | select((.basetrain == \"none\" or .basetrain ==\"no\") and .pretrain_iters!=\"0\" and (.subset | contains(\"all\"))) | select(.result)| select(.result_type | contains(\"finetune\"))] | max_by(.result | tonumber) | .result"
all_top_moco_moco="[.[] | select((.basetrain == \"moco_v2_800ep\") and .pretrain_iters!=\"0\" and (.subset | contains(\"all\"))) | select(.result)| select(.result_type | contains(\"finetune\"))] | max_by(.result | tonumber) | .result"
all_top_sup_moco="[.[] | select((.basetrain == \"imagenet_r50_supervised\") and .pretrain_iters!=\"0\" and (.subset | contains(\"all\"))) | select(.result)| select(.result_type | contains(\"finetune\"))] | max_by(.result | tonumber) | .result"
all_bn="[.[] | select(.result_type | contains (\"finetune\")) | select( .file | contains (\"bn\")) | select(.subset | contains(\"all\")) | select(.result)| select(.result_type | contains(\"finetune\"))] | max_by(.result | tonumber) | .result"


l1000_no_basetrain="[.[] | select( (.basetrain == \"none\" or .basetrain == \"no\") and .pretrain_iters==\"0\"  and (.subset | contains(\"1000\"))) | select(.result)| select(.result_type | contains(\"finetune\"))] | max_by(.result | tonumber) | .result"
l1000_base_moco="[.[] | select( (.basetrain == \"moco_v2_800ep\") and .pretrain_iters==\"0\"  and (.subset | contains(\"1000\"))) | select(.result)| select(.result_type | contains(\"finetune\"))] | max_by(.result | tonumber) | .result"
l1000_base_sup="[.[] | select( (.basetrain == \"imagenet_r50_supervised\") and .pretrain_iters==\"0\"  and (.subset | contains(\"1000\"))) | select(.result)| select(.result_type | contains(\"finetune\"))]  | max_by(.result | tonumber) | .result"

l1000_top_none_moco="[.[] | select((.basetrain == \"none\" or .basetrain ==\"no\") and .pretrain_iters!=\"0\" and (.subset | contains(\"1000\"))) | select(.result)| select(.result_type | contains(\"finetune\"))] | max_by(.result | tonumber) | .result"
l1000_top_moco_moco="[.[] | select((.basetrain == \"moco_v2_800ep\") and .pretrain_iters!=\"0\" and (.subset | contains(\"1000\"))) | select(.result)| select(.result_type | contains(\"finetune\"))] | max_by(.result | tonumber) | .result"
l1000_top_sup_moco="[.[] | select((.basetrain == \"imagenet_r50_supervised\") and .pretrain_iters!=\"0\" and (.subset | contains(\"1000\"))) | select(.result)| select(.result_type | contains(\"finetune\"))] | max_by(.result | tonumber) | .result"
l1000_bn="[.[] | select(.result_type | contains (\"finetune\")) | select( .file | contains (\"bn\")) | select(.subset | contains(\"1000\")) | select(.result)| select(.result_type | contains(\"finetune\"))] | max_by(.result | tonumber) | .result"


prefixes=(all l1000)
suffixes=(no_basetrain base_moco base_sup top_none_moco top_moco_moco top_sup_moco bn)

for data in ${datas[@]}; do

    echo "########################################################" $data
    result_file="./results/${data}_results.json"
    for prefix in ${prefixes[@]}; do
        for suffix in ${suffixes[@]}; do
            varname="${prefix}_${suffix}"
            command=${!varname}

            test_top1=$(jq "${!varname}" "$result_file")
            
            # echo "${!varname}" 

            echo -e $test_top1 "\t" $varname
        done
    done

done

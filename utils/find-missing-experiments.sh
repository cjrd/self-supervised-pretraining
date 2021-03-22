#!/usr/bin/env bash

export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

names=(bdd chest_xray_kids chexpert coco_2014 domain_net_clipart domain_net_infograph domain_net_painting domain_net_quickdraw domain_net_real domain_net_sketch resisc ucmerced)
basemodels=(moco_v2_800ep_basetrain no_basetrain imagenet_r50_supervised_basetrain)

for dname in "${names[@]}"
do
    # HPT pre-train have the same iters
    basemodels=("imagenet_r50_supervised_basetrain" "moco_v2_800ep_basetrain")

    # check HPT for different basetrained models
    for basem in "${basemodels[@]}"; do
         for iters in 0 50 500 5000 50000; do
            btmodel=$DIR/../OpenSelfSup/work_dirs/hpt-pretrain/$dname/$basem/$iters-iters/final_backbone.pth
            if [ ! -f "$btmodel" ] && [ ! "$iters" = "0" ] ; then
                echo "Missing:  $dname $basem: $iters"
            fi

            # now look for linear analyses
            loutput0=$DIR/../OpenSelfSup/work_dirs/hpt-pretrain/$dname/linear-eval/$basem/$iters-iters-linear-eval-lr-s0/iter_5000.pth
            loutput1=$DIR/../OpenSelfSup/work_dirs/hpt-pretrain/$dname/linear-eval/$basem/$iters-iters-linear-eval-lr-s1/iter_5000.pth
            loutput2=$DIR/../OpenSelfSup/work_dirs/hpt-pretrain/$dname/linear-eval/$basem/$iters-iters-linear-eval-lr-s2/iter_5000.pth

            if [ ! -f "$loutput0" ]; then
                echo "Missing LINEAR:  $dname $basem: $iters"
            fi

            if [ -v "${CHECK_MULTIPLE}" ]; then
                if [ ! -f "$loutput1" ] || [ ! -f "$loutput2" ] ; then
                    echo "Missing multiple linear samples:  $dname $basem: $iters"
                fi
            fi

         done
    done

    for iters in 0 5000 50000 100000 200000 400000; do
        btmodel=$DIR/../OpenSelfSup/work_dirs/hpt-pretrain/$dname/no_basetrain/$iters-iters/final_backbone.pth

        if [ ! -f "$btmodel" ] && [ ! "$iters" = "0" ]; then
            if [[ "$iters" != "400000" || ! "$dname" =~ ^(resisc|ucmerced) ]]; then
                echo "Missing:  $dname no_basetrain: $iters"
            else
                continue
            fi
        fi

        # now look for linear analyses
        loutput0=$DIR/../OpenSelfSup/work_dirs/hpt-pretrain/$dname/linear-eval/no_basetrain/$iters-iters-linear-eval-lr-s0/iter_5000.pth
        loutput1=$DIR/../OpenSelfSup/work_dirs/hpt-pretrain/$dname/linear-eval/no_basetrain/$iters-iters-linear-eval-lr-s1/iter_5000.pth
        loutput2=$DIR/../OpenSelfSup/work_dirs/hpt-pretrain/$dname/linear-eval/no_basetrain/$iters-iters-linear-eval-lr-s2/iter_5000.pth

        if [ ! -f "$loutput0" ]; then
            echo "Missing LINEAR:  $dname no_bastrain: $iters"
        fi

        if [ -v "${CHECK_MULTIPLE}" ]; then
            if [ ! -f "$loutput1" ] || [ ! -f "$loutput2" ] ; then
                echo "Missing multiple linear samples:  $dname no_basetrain: $iters"
            fi
        fi
    done


    ddir=$DIR/../OpenSelfSup/work_dirs/hpt-pretrain/$dname/linear-eval
    # echo $ddir
    # now we need to check the linear analysis, and number of samples for each of these...
    #

    # check the supervised init
    for iters in 0 50 500 5000 50000; do
        pth=$ddir/imagenet*/$iters/*/iter_5000.pth;
            find "${ddir}" -path "$pth" | while read -r lin_train_file; do
                echo $lin_train_file
            done
        # done
    done
done
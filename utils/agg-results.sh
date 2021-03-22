#!/usr/bin/env bash

# this script will aggregate all results in the work_dirs directory and print the results
if ! command -v jq &> /dev/null
then
    echo "jq could not be found: install with:"
    echo "conda install -c conda-forge jq"
    exit
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
rgx=""
if [[ ! -z "$1" ]]; then
    rgx="$1"
    echo "Using rgx $rgx"
fi
paths=("*${rgx}*linear-eval*.json" "*${rgx}*finetune/all-labels*json" "*${rgx}*finetune/1000-labels*json")
HPT_PREFIX="${DIR}/../OpenSelfSup/work_dirs/hpt-pretrain"

for pth in "${paths[@]}"; do
    # printf "\n\n %s \n\n" "$pth"
    find "${HPT_PREFIX}" -path "$pth" | while read -r train_log; do
        (
        log_name=$(echo "$train_log" | awk -F 'hpt-pretrain/' '{ print $NF }')
        val_top1=$(jq 'select(.["val-head0_top1"]) | .["val-head0_top1"]' "$train_log")
        test_top1=$(jq 'select(.["test-head0_top1"]) | .["test-head0_top1"]' "$train_log")

        readarray -t test_array <<< "$test_top1"

        # check for auroc scores
        if [[ ${#test_array[@]} -lt 2 ]]; then
            val_top1=$(jq 'select(.["val-auroc_0"]) | [. as $in| keys[] | select(contains("val-auroc"))| $in[.]] | add/length' "$train_log")
            test_top1=$(jq 'select(.["test-auroc_0"]) | [. as $in| keys[] | select(contains("test-auroc"))| $in[.]] | add/length' "$train_log")
            readarray -t test_array <<< "$test_top1"
        fi

        legacy=0
        if [[ ${#test_array[@]} -lt 2 ]]; then
            legacy=1
            # fallback to legacy: this may cause errors...
            test_top1=$(jq 'select(.["head0_top1"]) | .["head0_top1"]' "$train_log")
            readarray -t test_array <<< "$test_top1"
            if [[ ${#test_array[@]} -lt 2 ]]; then
	      test_top1=$(jq 'select(.["unnamed-val-hook-head0_top1"]) | .["unnamed-val-hook-head0_top1"]' "$train_log")
              readarray -t test_array <<< "$test_top1"
	      if [[ ${#test_array[@]} -lt 2 ]]; then
		  >&2 echo "WARNING: will not include unparseable result for ${HPT_PREFIX}${log_name}"		
                  continue
	      fi
            fi

            # use the text log to get val scores
            logdir="$(dirname $train_log)"
            logname=$(basename ${train_log})
            log_file=${logname%.json}
            val_log_name="$logdir/train_$log_file"
	    # last tail call removes the first evaluation, which isn't present in the json
            val_top1=$(cat "$val_log_name" | grep " - head0_top1" | awk 'NR % 2' | awk '{ print $NF }' | tail -n +2)
        fi

        if [[ ${#test_array[@]} -lt 20 ]]; then
	        # check if process file exists
	        procfile=$(dirname "${HPT_PREFIX}${log_name}")".proc"
            if [[ "${MOVE_INCOMPLETE}" == "true" && ! -f "${procfile}" ]]; then
		        mkdir -p ${DIR}/incomplete_results
                rmdir=$(dirname "${log_name}")
		        >&2 echo "moving incomplete ${HPT_PREFIX}./${rmdir} to ${DIR}/incomplete_results"
		        rsync -a --remove-source-files --relative ${HPT_PREFIX}./${rmdir} ${DIR}/incomplete_results
	        elif [[ "${MOVE_INCOMPLETE}" == "true" ]]; then
		        >&2 echo "will not include partial result for ${HPT_PREFIX}${log_name}: procfile exists so will not remove"
	        else
		        >&2 echo "will not include partial result for ${HPT_PREFIX}${log_name}"
            fi
            exit 1
        fi


        # pull out the top test value based on the top validation value
        max_val=0
        ct=0
        vallen=$(echo ${val_top1} | wc -w)
        # only take best result in last 1/3
        afterlen=$(echo "$vallen * 0.5" | bc -l | xargs printf "%0.0f" )
        for val in $val_top1; do
            if (( $(echo "$val > $max_val" |bc -l) )); then
                if [[ $ct > $afterlen ]]; then
                    max_val=$val
                    test_val=${test_array[ct]}
                fi
            fi
            ((ct++))
        done

        if [[ $legacy == 0 ]]; then
            echo "${log_name} ::: $test_val"
        else
            echo "${log_name} ::: $test_val [LEGACY FORMAT PARSED - DOUBLE CHECK TENSORBOARD]"
        fi
	)& # trivially parallelize this function
	(( i++ % 200 == 0 )) && wait
    done
done

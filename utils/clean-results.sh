#!/usr/bin/env bash


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"



function archive_duplicate_logs () {
        # find experiments with multiple results jsons
        num_res=$(ls ${train_dir}/*json | wc -l)
        if [[ $num_res > 1 ]]; then
            echo "Multiple logs: ${work_dirs}/${train_dir}"
            max_score=-1
            max_file=""
            for lfile in ${train_dir}/*json; do
                score=$(jq -r '.epoch' ${lfile} | sort -n | tail -n 1)
                if [[ $score > $max_score  && "$score" != "null" ]]; then
                    if [[ "${max_file}" != "" ]]; then
                        mv ${max_file} ${max_file}-ARCHIVE
                    fi
                    max_file=$lfile
                    max_score=$score
                elif [[ $score == $max_score ]]; then
                    # take the newest file
                    max_date=$(echo $(basename $max_file) | awk -F '_' '{ print $1 }')
                    lfile_date=$(echo $(basename $lfile) | awk -F '_' '{ print $1 }')
                    if [[ $lfile_date > $max_date ]]; then
                        mv $max_file $max_file-ARCHIVE
                        max_file=$lfile
                    else
                        mv $lfile $lfile-ARCHIVE
                    fi
                else
                    mv $lfile $lfile-ARCHIVE
                fi
            done
            echo "Max log: ${work_dirs}/${lfile}"
        fi
}


paths=("*linear-eval*.json" "*finetune/all-labels*json" "*finetune/1000-labels*json")
HPT_PREFIX="../OpenSelfSup/work_dirs/hpt-pretrain/"

archive_dir="${DIR/../OpenSelfSup/work_dirs/archive_incomplete_results}"
echo "Archiving to ${archive_dir}"

linear_complete_file="iter_5000.pth"
work_dirs=${DIR}/../OpenSelfSup/work_dirs/hpt-pretrain
cd $work_dirs || exit 1
for pth in "${paths[@]}"; do
    find . -path "$pth" | while read -r train_log; do
        ###########################
        # check if procfile exists
        ###########################
        # TODO add age check on proc file 
        train_dir=$(dirname ${train_log})

        proc_file="${train_dir}.proc"
        if [[ -f "${proc_file}" ]]; then
            echo "Proc file exists - ${work_dirs}/${proc_file}, will not move"
            continue
        fi
        
        
        ###############
        # handle linear
        ###############

        if [[ "$pth" == "*linear-eval*.json" ]]; then
            complete_file="${work_dirs}/${train_dir}/${linear_complete_file}"
            if [[ ! -f "$complete_file" ]]; then
                echo "missing ${complete_file}"
                rsync -a --remove-source-files --relative ./${train_dir} ../hpt-incomplete-results
            fi

            num_res=$(ls ${train_dir}/*json | wc -l)
            if [[ $num_res > 1 ]]; then
                echo "Multiple logs: ${work_dirs}/${train_dir}"
            fi

            archive_duplicate_logs

            continue
        fi

        #####################
        # handle finetuning
        #####################
        #echo ${work_dirs}$train_log

        # name of finetuning dirs are of the type X-iters-Y-Z
        # we need Y-Z to check for final it Z_Y.pth
        final_file=$(basename $(dirname $train_log) | awk -F '-' '{print $(NF-3)"_"$(NF-4)".pth"}')
        full_final_file="${train_dir}/${final_file}"

        if [[ ! "$final_file" =~ "epoch" && ! "$final_file" =~ "iter" ]]; then
            echo "Result file does not match expected format, will not move: ${work_dirs}/${full_final_file}"
        fi

        if [[ ! -f "$full_final_file" ]]; then
            # 2500 iter did not checkpoint at the end =\
            if [[ "$final_file" != "iter_2500.pth" ]]; then
                echo "Missing. Please investigate manually: ${work_dirs}/${full_final_file}"
            fi
        fi

        archive_duplicate_logs

    done
done


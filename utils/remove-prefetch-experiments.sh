#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

backupdir=$1

cd $DIR/../OpenSelfSup/work_dirs/hpt-pretrain
# remove the files
rg 'prefetch *= *True' | head | awk -F ":" '{print $1}' | rev | cut -d/ -f "2-" | rev | sort | uniq | while read -r baddir; do
    echo $baddir
    # remove the actual run files (back them up for now to $1)
    # rsync -a --remove-source-files --relative ././$baddir $backupdir
done



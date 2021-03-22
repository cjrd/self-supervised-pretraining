#!/usr/bin/env bash
set -e

# this script will aggregate all results in the work_dirs directory and print the results
if ! command -v jq &> /dev/null
then
    echo "jq could not be found: install with e.g. (or use pip, etc (e.g. google it)):"
    echo "conda install -c conda-forge jq"
    exit
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TMPDIR="${DIR}"/tmp
TMPJSONDIR="${TMPDIR}"/jsons
TMPRES="${TMPDIR}/results.txt"
ERRORS="${TMPDIR}/errors.txt"
FINAL_RESULTS_DIR="${DIR}/../results"

rm -rf "${TMPDIR}"
mkdir -p "$TMPDIR"
mkdir -p "$FINAL_RESULTS_DIR"

# first aggregate all the results
"${DIR}"/agg-results.sh "$1" 2> ${ERRORS} | sort -V > "$TMPRES"

# create json
python "${DIR}"/agg-results-to-final-json.py --agg-file "${TMPRES}" --outdir "${TMPJSONDIR}"

# now merge with existing jsons (if they exist)
find "${TMPJSONDIR}" -name '*.json' | while read -r resname; do
    output=${FINAL_RESULTS_DIR}/$(basename $resname)
    if [[ -f "$output" ]]; then
        echo "Merging results into $output"
        jq -s -S '.[0] * .[1]' "$output" "$resname" > tempfile
        mv tempfile "$output"
    else
        echo "Sending results into $output"
        jq -S "." "$resname" > "$output"
    fi
    sed -i 's/"basetrain": "imagenet_r50_21352794",/"basetrain": "imagenet_r50_supervised",/g' "$output"
done

if [ -s "$ERRORS" ]; then
    RED='\033[0;31m'
    NC='\033[0m' # No Color
    printf "\n\n${RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\nPlease investigate as your results may not be complete.\n"
    printf "(see errors in file: ${ERRORS})\n\n"
    cat "$ERRORS"
    printf "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${NC}\n"   

    error_files=$(cat "$ERRORS" | awk -F 'hpt-pretrain/' '{print $NF}')
    for e in $error_files; do
        baseres=$(echo "$e" | awk -F '/' '{print $1}')
        # now remove these from the results files incase they were previously added
        resname="${FINAL_RESULTS_DIR}/${baseres}_results.json"
        if [ -f "$resname" ]; then
            jq -r "del(.[\"${e}\"])" "${resname}" > tmp.json
            mv tmp.json ${resname}
        fi
    done
fi

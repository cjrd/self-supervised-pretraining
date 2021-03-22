#!/bin/bash

bash utils/finetune_compare_exp_results.sh > .tmp_result.txt
python utils/plot-results-exp-2.py
echo "Done with plots: ./plot-results/exp-2/*.pdf"

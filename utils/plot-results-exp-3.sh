#!/bin/bash

bash utils/hpt_framework_analysis.sh > .tmp_exp3_result.txt
python utils/plot-results-exp-3.py
echo "Done with plots: ./plot-results/exp-3/*.pdf"

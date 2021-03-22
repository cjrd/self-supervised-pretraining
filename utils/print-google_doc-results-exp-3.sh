#!/bin/bash

bash utils/hpt_framework_analysis_full_print.sh > .aaa.aaa
python utils/print-google_doc-results-exp-3.py > results/google_doc_result.txt

echo "Result in: results/google_doc_result.txt"

#!/usr/bin/env bash

# Stop if any command fails (optional, but handy)
set -e

python ../unsupervised-modeling/accuracy.py milas_grammars/results/productions_NVA_VB_pb0p1_lb0p01_primed_20251128_1529/test_tags.txt ../../Downloads/all_tags.txt --mode one2one --tokens-file milas_grammars/test_en_cleaned.txt 

python ../unsupervised-modeling/accuracy.py milas_grammars/results/productions_NVA_VB_pb0p1_lb0p1_primed_20251128_1542/test_tags.txt ../../Downloads/all_tags.txt --mode one2one --tokens-file milas_grammars/test_en_cleaned.txt 

python ../unsupervised-modeling/accuracy.py milas_grammars/results/productions_NVA_VB_pb0p1_lb0p1/test_tags.txt ../../Downloads/all_tags.txt --mode one2one --tokens-file milas_grammars/test_en_cleaned.txt 

python ../unsupervised-modeling/accuracy.py milas_grammars/results/productions_NVA_VB_pb0p1_lb0p01/test_tags.txt ../../Downloads/all_tags.txt --mode one2one --tokens-file milas_grammars/test_en_cleaned.txt 



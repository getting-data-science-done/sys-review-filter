#!/bin/bash
#
# Multiple steps required to prepare all data for experiments
#
# 1. Join with questions into complete dataframes

scripts/JOIN_DATA_TO_QUESTIONS.sh

# 2. Apply Feature Engineering and cleaning to data.
#    NOTE: These script can take many hours to complete
# 
#    2.a First the basic question features 

python3 scripts/prepare_features.py data/all_data.csv Title Abstract Authors Question data/processed.csv

python3 scripts/clean_dataset.py data/processed.csv


#    2.b Then the PICO features

python3 scripts/prepare_pico_features.py data/all_data_pico.csv Title Abstract Authors Question data/processed_pico.csv

python3 scripts/clean_dataset.py data/processed_pico.csv


#    2.c Then the BERT similarity features

python3 scripts/sentence_similarity_features.py data/processed.csv biobert TEXT Question data/similarity_processed.csv

python3 scripts/sentence_similarity_features.py data/similarity_processed.csv bluebert TEXT Question data/similarity_processed_v2.csv

python3 scripts/sentence_similarity_features.py data/similarity_processed.csv bluebertxl TEXT Question data/similarity_processed_v3.csv

python3 scripts/pico_similarity_features.py data/processed_pico.csv bluebert TEXT Question data/pico_similarity.csv



# 3. Finally make all those datasets available via projit

projit init sys_review_filter
projit add dataset train data/processed.csv
projit add dataset train_pico data/processed_pico.csv
projit add dataset similarity data/similarity_processed.csv
projit add dataset similarity_v2 data/similarity_processed_v2.csv
projit add dataset similarity_v3 data/similarity_processed_v3.csv
projit add dataset pico_similarity data/pico_similarity.csv



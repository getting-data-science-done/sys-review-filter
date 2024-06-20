#!/bin/bash

# Simple ML Approaches

python3 experiments/01_1_Baseline.py
python3 experiments/01_2_ExtraTrees.py
python3 experiments/01_3_ExtraTrees_TFIDF.py
python3 experiments/01_4_LGBM_TFIDF.py

# Repeated using PICO framework questions

python3 experiments/02_1_Baseline_PICO.py
python3 experiments/02_2_ExtraTrees_PICO.py
python3 experiments/02_3_ExtraTrees_TFIDF_PICO.py
python3 experiments/02_4_LGBM_TFIDF_PICO.py

# Including BERT based text similarity features
python3 experiments/03_1_LGBM-BioBERT.py
python3 experiments/03_2_LGBM-BlueBERT.py
python3 experiments/03_3_LGBM-BlueBERTxL.py
python3 experiments/03_4_LGBM-BlueBERT-PICO.py

# Finally fine-tuning BERT models for direct prediction

python3 experiments/04_1_Epoch_Finetune_BioBertPubMed.py
python3 experiments/04_2_Epoch_Finetune_BioBertPubMed.py
python3 experiments/04_3_Epoch_Finetune_BioBertPubMed.py


#!/bin/bash
#
# Once we have the individual datasets for each study
# we need to get them into a single CSV with their appropriate
# question for the study.
#
# NOTE: This script should be executed from the root directory
#

python3 scripts/join_with_questions.py _Question.txt data/all_data.csv
python3 scripts/join_with_pico.py  _Question.txt data/all_data_pico.csv


# Systematic Review Filter

[![paper](https://img.shields.io/badge/arxiv-2405.20354-b31b1b)](https://arxiv.org/abs/2405.20354)
[![page views count](https://badges.toozhao.com/badges/01HYS7AXHZQ8HX64H1K1F3379K/blue.svg)](https://github.com/getting-data-science-done/sys-review-filter)

This repository contains the code and data for the paper 'Literature Filtering for Systematic Reviews with Transformers'

## Setup

The required libraries for this project can be initialised using the script
[RUN_01_SETUP.sh](RUN_01_SETUP.sh)


## Data

You can see each of the research questions inside an individual folder within
the [data](data) directory. This data  was kindly provided by the Royal Australian College
of Surgeons systematic review team ASERNIP-S.

Processing of this data to prepare it for all machine learning experiments is done
with the data preparation script: [RUN_02_PREPARE_DATA.sh](RUN_02_PREPARE_DATA.sh)

## Experiments

All experiments are written as single executable python 3 scripts in the 
directory [experiments](experiments). Note that these experiments require
the [projit](https://pypi.org/project/projit/) 
library to both access the shared dataset and to stores
results and meta-data about the experiment.

You can execute all experiments using the experiment run master script: 
[RUN_03_EXPERIMENTS.sh](RUN_03_EXPERIMENTS.sh) 


## Results

The results directory contains some of the summarised results use in the
publication.




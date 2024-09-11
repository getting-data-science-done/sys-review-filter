# Systematic Review Filter

[![paper](https://img.shields.io/badge/arxiv-2405.20354-b31b1b)](https://arxiv.org/abs/2405.20354)

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

If you use the data or code from this repository in your research please
cite the conference paper:

```bibtex
@inproceedings{hawkins+tivey:2024,
 author = {John Hawkins and David Tivey},
 year = {2024},
 month = {06},
 title = {Literature Filtering for Systematic Reviews with Transformers},
 booktitle = {2nd International Conference on Communications, Computing and Artificial Intelligence (CCCAI 2024)},
 address = {Jeju, Korea},
 editor = {},
 ISBN = {},
 doi = {https://doi.org/10.1145/3676581.3676582}
}
```



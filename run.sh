#!/bin/bash

module load anaconda/2021.05
module load cuda/11.3
module load gcc/7.3

source activate openmmlab

export PYTHONUNBUFFERED=1

python tools/train.py \
    configs/work1_config_used/work1_twins_svt_base_1xb64_flower5.py \
    --work-dir work/work1_twins_1xb64_flower5_top1

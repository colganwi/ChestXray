#!/bin/bash

export PYTHONPATH=$PYTHONPATH:"/sonigroup/ChestXray/code/"
export PYTHONPATH=$PYTHONPATH:"/sonigroup/ChestXray/models/research/slim/"

DATASET_NAME=chestXray
DATASET_DIR=/sonigroup/ChestXray/data/
TRAIN_DIR=/sonigroup/ChestXray/checkpoints/split1_np_densenet2/

python code/eval_localizer.py \
    --alsologtostderr \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --eval_dir=${TRAIN_DIR} \
    --dataset_split_name=split_1/test_bbox \
    --model_name=np_densenet121 \

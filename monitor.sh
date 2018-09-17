#!/bin/bash

export PYTHONPATH=$PYTHONPATH:"/sonigroup/ChestXray/code/"
export PYTHONPATH=$PYTHONPATH:"/sonigroup/ChestXray/models/research/slim/"

DATASET_NAME=chestXray
DATASET_DIR=/sonigroup/ChestXray/data/
CHECKPOINT_DIR=/sonigroup/ChestXray/checkpoints/
declare -a models=("split1_np_densenet1" "split1_np_densenet2")

for i in {1..2}
do
  for j in "${models[@]}"
  do
    python code/eval_classifier.py \
        --alsologtostderr \
        --checkpoint_path=$CHECKPOINT_DIR$j \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=${DATASET_NAME} \
        --eval_dir=$CHECKPOINT_DIR$j \
        --dataset_split_name=split_1/test \
        --model_name=densenet121
    python code/eval_localizer.py \
        --alsologtostderr \
        --checkpoint_path=$CHECKPOINT_DIR$j \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=${DATASET_NAME} \
        --eval_dir=$CHECKPOINT_DIR$j \
        --dataset_split_name=split_1/test_bbox \
        --model_name=densenet121
  done
  sleep 3600
done

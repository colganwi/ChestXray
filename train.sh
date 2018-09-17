#!/bin/bash

export PYTHONPATH=$PYTHONPATH:"/sonigroup/ChestXray/code/"
export PYTHONPATH=$PYTHONPATH:"/sonigroup/ChestXray/models/research/slim/"

MODEL_NAME=np_densenet121
DATASET_NAME=chestXray
DATASET_DIR=/sonigroup/ChestXray/data/
TRAIN_DIR=/sonigroup/ChestXray/checkpoints/split1_np_densenet2/
PRETRAINED_PATH=/sonigroup/ChestXray/pretrained/densenet121/tf-densenet121.ckpt

python code/train_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --checkpoint_path=${PRETRAINED_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=split_1/train \
    --model_name=${MODEL_NAME} \
    --weighted_loss=True \
    --learning_rate=.001 \
    --optimizer=adam \
    --batch_size=8 \
    --num_epochs_per_decay=10 \
    --learning_rate_decay_factor=.1 \
    --checkpoint_exclude_scopes=densenet121/logits \
    --max_number_of_steps=150000 \
    --localization_loss=True

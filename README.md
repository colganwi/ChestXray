# ChestXray

This repository implements a 121-layer [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) which achieved AUROC values comparable to the state of the art for the [ChestXray14](https://arxiv.org/pdf/1705.02315.pdf) dataset. You can read my paper [here](https://github.com/colganwi/ChestXray/blob/master/paper/paper.pdf).

<img src="https://github.com/colganwi/ChestXray/blob/master/paper/figure_5.png" width="480">
Class activation maps generated by the 121-layer DenseNet. <br><br>

The framework is borrowed from [TensorFlow-Slim Models](https://github.com/tensorflow/models/tree/master/research/slim)

## Usage

### Instalation
1. Follow the instructions for [TensorFlow-Slim Models](https://github.com/tensorflow/models/tree/master/research/slim)
2. Download the ChestXray14 dataset from [Box](https://nihcc.app.box.com/v/ChestXray-NIHCC)
3. Set paths for images and labels in code/make_TFRecord.py.
4. Make TFRecord file. This will take a few hours.
```
$ python code/make_TFRecord.py
```
### Training from scratch
```
$ MODEL_NAME=densenet121
$ DATASET_NAME=chestXray
$ DATASET_DIR=/ChestXray/data/
$ TRAIN_DIR=/ChestXray/checkpoints/densenet/
$ python code/train_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=standard/train \
    --model_name=${MODEL_NAME} \
    --weighted_loss=True \
    --learning_rate=.001 \
    --optimizer=adam \
    --batch_size=16 \
    --num_epochs_per_decay=10 \
    --learning_rate_decay_factor=.1 \
 ```
 ### Fine-tuning from existing checkpoint
 Download the checkpoint file for DenseNet-121 from [here](https://drive.google.com/open?id=0B_fUSpodN0t0eW1sVk1aeWREaDA).
 ```
$ MODEL_NAME=densenet121
$ DATASET_NAME=chestXray
$ DATASET_DIR=/ChestXray/data/
$ TRAIN_DIR=/ChestXray/checkpoints/densenet/
$ PRETRAINED_PATH=/ChestXray/checkpoints/tf-densenet121.ckpt
$ python code/train_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --checkpoint_path=${PRETRAINED_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=standard/train \
    --model_name=${MODEL_NAME} \
    --checkpoint_exclude_scopes=densenet121/logits \
    --weighted_loss=True \
    --learning_rate=.001 \
    --optimizer=adam \
    --batch_size=16 \
    --num_epochs_per_decay=10 \
    --learning_rate_decay_factor=.1 \
 ```
 ### Evaluating model
 ```
$ DATASET_NAME=chestXray
$ DATASET_DIR=/ChestXray/data/
$ TRAIN_DIR= /ChestXray/checkpoints/densenet/

$ python code/eval_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --eval_dir=${TRAIN_DIR} \
    --dataset_split_name=standard/val \
    --model_name=densenet121 \
```
## Results

### Optimization
I experimented with different optimzers, learning rate decay schedules, and data augmentation strategies.

<img src="https://github.com/colganwi/ChestXray/blob/master/paper/figure_3A.png" width="480">

The weighted cross entropy loss (W_CEL) generally plateaued after about 100,000 steps. Both Adam and Momentum were effective as long as the learning rate decay schedule and data augmentation strategy prevented overfitting.

### Integrating localization
The chestXray14 dataset contains bounding boxes for about 1000 findings. I hypothesized that including localization in training would increase both localization and classification accuracy since the localization is predictive.

<img src="https://github.com/colganwi/ChestXray/blob/master/paper/figure_6.png" width="480">
Spatial distribution of bounding boxes across different diseases. <br><br>

To achieve this I weakly supervised with the bounding boxes. I did this by adding the W_CEL between the class activation map  and the bounding box to the loss function. This can be done with localization loss flag.
```
--localization_loss=True
```
This increased the AUROC and IoU values.

### Increasing resultion
The prediction layer for DenseNet-121 is 7x7 which was not high enough resolution for many diseases. The fix this issue I removed the 2x2 average pooling layers.
```
$ MODEL_NAME=np_densenet121
```
This modification further increases the IoU and AUROC. The AUROC values are comparable to the state of the art achieved by [Rajpurkar, et al.](https://arxiv.org/pdf/1711.05225.pdf).

<img src="https://github.com/colganwi/ChestXray/blob/master/paper/figure_4.png" width="480">

The localization is good for some findings like Cardiomegaly but still need improvement for small findings like Mass

<img src="https://github.com/colganwi/ChestXray/blob/master/paper/figure_7.png" width="480">

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the ChestXray 2017 Dataset. See details here:
http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf
ChestXray8 2017 contains 121,210 frontal chest x-rays with 8 NLP generated
lables. It also contains 1000 bounding boxes. The dataset is split into a training
set (70%), validation set (10%), test set 1 (10%), and test set 2 (10%)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tensorflow as tf
import chestXray_decoder

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s.tfrecord'

_SPLITS_TO_SIZES = {
    'standard/train': 74903,
    'standard/val': 10621,
    'standard/test1': 12877,
    'standard/test2': 12719,
    'split_1/train' : 89669,
    'split_1/test' : 22451,
    'split_1/test_bbox' : 181,
    'sample/sample' : 10000

}

_ITEMS_TO_DESCRIPTIONS = {
    'image/height': 'hight',
    'image/width': 'width',
    'image/encoded': 'grayscale image of size hight x width',
    'image/view':  'AP or PA',
    'image/follow_up': 'The follow up number when the x-ray was taken',
    'iamge/patient/age': 'age',
    'image/patient/gender': 'M or F',
    'image/finding': '1:finding, 0:no finding',
    'image/finding/class/text': 'finding text',
    'image/finding/class/label': '[0,1,2,3,4,5,6,7,8]',
    'image/object/class/text': 'object text',
    'image/object/class/label': '[0,1,2,3,4,5,6,7,8]'
}

_CLASSES = range(1,15)
_NUM_CLASSES = len(_CLASSES)


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading chestXray.
  Args:
    split_name: A split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.
  Returns:
    A `Dataset` namedtuple.
  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
    'image/filename': tf.FixedLenFeature(
      (), dtype=tf.string),
    'image/encoded': tf.FixedLenFeature(
      (), dtype=tf.string),
    'image/format': tf.FixedLenFeature(
      (), dtype=tf.string),
    'image/view': tf.FixedLenFeature(
      (), dtype=tf.string),
    'image/follow_up': tf.FixedLenFeature(
      (), dtype=tf.int64),
    'iamge/patient/age': tf.FixedLenFeature(
      (), dtype=tf.int64),
    'image/patient/gender': tf.FixedLenFeature(
      (), dtype=tf.string),
    'image/findings': tf.VarLenFeature(
      dtype=tf.int64),
    'image/bboxs': tf.VarLenFeature(
        dtype=tf.int64),
    'image/bboxs/encoded': tf.FixedLenFeature(
       (), dtype=tf.string),
    'image/bboxs/format': tf.FixedLenFeature(
       (), dtype=tf.string),
  }

  items_to_handlers = {
      'image': chestXray_decoder.Image('image/encoded', 'image/format',channels=1,shape=[512,512]),
      'label': chestXray_decoder.Tensor('image/findings',shape=[14]),
      'location': chestXray_decoder.Image('image/bboxs/encoded',
         'image/bboxs/format',channels=1,shape=[50176,14]),
      'bbox': chestXray_decoder.Tensor('image/bboxs',shape=[14])
  }

  decoder = chestXray_decoder.ChestXrayDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)

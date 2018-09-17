from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import dataset_factory
import net_factory
import preprocess_factory

slim = tf.contrib.slim

MODEL_NAME = 'densenet121'
BATCH_SIZE = 8
#with tf.Graph().as_default():
tf_global_step = slim.get_or_create_global_step()

######################
# Select the dataset #
######################
dataset = dataset_factory.get_dataset(
    'chestXray','sample/sample','/sonigroup/ChestXray/data/')

####################
# Select the model #
####################
network_fn = net_factory.get_network_fn(
    MODEL_NAME,
    num_classes=dataset.num_classes,
    is_training=False)

##############################################################
# Create a dataset provider that loads data from the dataset #
##############################################################
provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    shuffle=False,
    common_queue_capacity=2 * BATCH_SIZE,
    common_queue_min=BATCH_SIZE)

[image] = provider.get(['image'])
image = tf.stack([image,image,image],axis=2)


#####################################
# Select the preprocessing function #
#####################################

image_preprocessing_fn = preprocess_factory.get_preprocessing(
    MODEL_NAME,is_training=False)

eval_image_size = 224
image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

'''
images, labels  = tf.train.batch(
    [image],
    batch_size=BATCH_SIZE,
    num_threads=4,
    capacity=5 * BATCH_SIZE)
'''

iamge = tf.constant(5)

####################
# Define the model #
####################
#logits, endpoints = network_fn(images)


sess = tf.Session()
with sess.as_default():
  print(image.eval())

sess.close()

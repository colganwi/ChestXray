import tensorflow as tf
import numpy as np
from load_data import *
import six
import sys
import time
import PIL

sys.path.append('/sonigroup/ChestXray/models/research')

from object_detection.utils import dataset_util
from object_detection.utils import visualization_utils
from PIL import Image
SPLIT = 'sample/sample'

flags = tf.app.flags
FLAGS = flags.FLAGS


def create_tf_example(example):
  #label map
  MAP = {'Atelectasis':0,'Cardiomegaly':1,'Effusion':2,
  'Infiltrate':3,'Infiltration':3,'Mass':4,'Nodule':5,'Pneumonia':6,'Pneumothorax':7,
  'Consolidation':8,'Edema':9,'Emphysema':10,'Fibrosis':11,
  'Pleural_Thickening':12,'Hernia':13
  }

  # TODO(user): Populate the following variables from your example.
  height = 1024 # Image height
  width = 1024 # Image width
  filename = example[0] # Filename of the image. Empty if image is not from file
  img = Image.open("/sonigroup/ChestXray/data/images/"+example[0])
  img = img.resize((512,512),PIL.Image.BILINEAR)
  img = img.convert(mode='L')
  output = six.BytesIO()
  img.save(output, format='PNG')
  encoded_image_data = output.getvalue()
  image_format = b'png'
  view = example[1]['View Position']
  follow_up = int(example[1]['Follow-up #'])
  age = int(example[1]['Patient Age'])
  gender = example[1]['Patient Gender']
  findings_text = example[1]['Finding Labels'].split('|')
  if findings_text[0] == 'No Finding':
      finding = 0
  else:
      finding = 1
  findings = np.zeros((14),np.uint8)
  for a in findings_text:
      if MAP.has_key(a):
          findings[[MAP[a]]] = 1

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  for a in example[2]:
      xmins += [float(a['Bbox [x'])/1024]
      ymins += [float(a['y'])/1024]

      xmaxs += [(float(a['Bbox [x'])+float(a['w']))/1024]
      ymaxs += [(float(a['y'])+float(a['h]']))/1024]

      classes_text += [a['Finding Label']]
      if MAP.has_key(a['Finding Label']):
          classes += [MAP[a['Finding Label']]]

  bboxs = np.zeros([224,224,14],np.uint8)
  loc_bboxs = [0]*14
  for i in range(len(xmins)):
      bboxs[int(ymins[i]*223):int(ymaxs[i]*223),
            int(xmins[i]*223):int(xmaxs[i]*223),classes[i]] = 1
      loc_bboxs[classes[i]] = 1
  bboxs = np.reshape(bboxs,[50176,14])
  bboxs = Image.fromarray(bboxs,mode='L')
  output = six.BytesIO()
  bboxs.save(output, format='PNG')
  encoded_bboxs_data = output.getvalue()
  bboxs_format = b'png'


  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/filename': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/view':  dataset_util.bytes_feature(view),
      'image/follow_up': dataset_util.int64_feature(follow_up),
      'iamge/patient/age': dataset_util.int64_feature(age),
      'image/patient/gender': dataset_util.bytes_feature(gender),
      'image/findings': dataset_util.int64_list_feature(findings),
      'image/bboxs': dataset_util.int64_list_feature(loc_bboxs),
      'image/bboxs/encoded': dataset_util.bytes_feature(encoded_bboxs_data),
      'image/bboxs/format': dataset_util.bytes_feature(bboxs_format),
  }))
  return tf_example


def main(_):

  writer = tf.python_io.TFRecordWriter("/sonigroup/ChestXray/data/"+SPLIT+".tfrecord")

  # TODO(user): Write code to read in your dataset to examples variable


  list = load_list("/sonigroup/ChestXray/data/"+SPLIT+"_list.txt")
  data = load_csv("/sonigroup/ChestXray/data/Data_Entry_2017.csv")
  bbox = load_csv("/sonigroup/ChestXray/data/BBox_List_2017.csv")

  examples = []
  for a in list:
      if bbox.has_key(a):
          examples += [[a,data[a][0],bbox[a]]]
      else:
          examples += [[a,data[a][0],[]]]

  j = 0
  for example in examples:
    j += 1
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())
    if j%1000 == 0:
        print(str(j)+" images done")

  writer.close()


if __name__ == '__main__':
  tf.app.run()

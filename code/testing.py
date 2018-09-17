import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from load_data import *
import random
import h5py
slim = tf.contrib.slim

#label map
MAP = {'Atelectasis':0,'Cardiomegaly':1,'Effusion':2,
'Infiltration':3,'Mass':4,'Nodule':5,'Pneumonia':6,'Pneumothorax':7,
'Consolidation':8,'Edema':9,'Emphysema':10,'Fibrosis':11,
'Pleural_Thickening':12,'Hernia':13
}

"""
labels = tf.cast(tf.constant([1,0,1]),tf.bool)
preds = tf.cast(tf.constant([1,1,1]),tf.bool)
out = tf.reduce_sum(tf.cast(tf.logical_or(preds,labels),tf.float16))

sess = tf.Session()
ret = sess.run(out)
print(ret)
sess.close()
"""

"""
MAP = {'No Finding':0,'Atelectasis':1,'Cardiomegaly':2,'Effusion':3,
    'Infiltrate':4,'Mass':5,'Nodule':6,'Pneumonia':7,'Pneumothorax':8,
    'Consolidation':9,'Edema':10,'Emphysema':11,'Fibrosis':12,
    'Pleural_Thickening':13,'Hernia':14
    }
"""
"""
locations = []
for i in range(14):
    locations += [np.zeros([1024,1024])]

bbox = load_csv("/sonigroup/ChestXray/data/BBox_List_2017.csv")
for key in bbox.keys():
    box = np.zeros([1024,1024])
    x_start = int(float(bbox[key][0]['Bbox [x']))
    y_start = int(float(bbox[key][0]['y']))
    x_end = x_start + int(float(bbox[key][0]['w']))
    y_end = y_start + int(float(bbox[key][0]['h]']))
    box[x_start:x_end,y_start:y_end] = 1
    locations[MAP[bbox[key][0]['Finding Label']]-1] += box

for i in range(14):
    image = locations[i]
    image = np.swapaxes(image,0,1)
    image = image/np.max(image)
    img = Image.fromarray(np.uint8(cm.jet(image)*255))
    img.save(str(i)+'_bbox.png')
"""
'''
images = load_csv("/sonigroup/ChestXray/data/Data_Entry_2017.csv")
bboxs = []
for i in range(5):
    z = load_list("/sonigroup/ChestXray/data/split_"+str(i+1)+"/test_bbox_list.txt")
    bboxs += [[x.split('_')[0] for x in z]]


#list = load_list("/sonigroup/ChestXray/data/bbox_test/test1_list.txt")
lists = [[],[],[],[],[]]
nums = np.zeros((14,5))
last_patientID = ''
last_list = 0

x = list(images.keys())
x.sort()

for key in x:
    label = images[key][0]['Finding Labels'].split('|')[0]
    if label == 'No Finding':
        foo = range(5)
        i = random.choice(foo)
    else:
        i = MAP[images[key][0]['Finding Labels'].split('|')[0]]
    patientID = key.split('_')[0]
    if patientID == last_patientID:
        lists[last_list] += [key]
        nums[i][last_list] += 1
    else:
        last_patientID = patientID
        list = None
        for j in range(5):
            if patientID in bboxs[j]:
                list = j
        if list == None:
            list = np.argmin(nums[i])
        lists[list] += [key]
        nums[i][list] += 1
        last_list = list

print(nums)


for i in range(5):
   save_list("/sonigroup/ChestXray/data/split_"+str(i+1)+"/test_list.txt",lists[i])
   save_list("/sonigroup/ChestXray/data/split_"+str(i+1)+"/train_list.txt",[x for x in images.keys() if x not in lists[i]])
'''

f = h5py.File('trained_densenet121_features.h5','r')
print(f.keys())
data = f.get('global_pool') # Get a certain dataset
data = np.array(data)
print(data)

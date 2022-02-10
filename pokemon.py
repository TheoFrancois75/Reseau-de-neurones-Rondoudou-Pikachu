# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:47:17 2022

@author: theo-
"""
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import datetime
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf

import pathlib
import os


# Recuperation des array des  images
'''
url_pikachu = r'https://github.com/anisayari/Youtube-apprendre-le-deeplearning-avec-tensorflow/blob/master/%234%20-%20CNN/pikachu.png?raw=true'
resp = requests.get(url_pikachu, stream=True).raw
image_array_pikachu = np.asarray(bytearray(resp.read()), dtype="uint8")

url_rondoudou = r'https://github.com/anisayari/Youtube-apprendre-le-deeplearning-avec-tensorflow/blob/master/%234%20-%20CNN/rondoudou.png?raw=true'
resp = requests.get(url_rondoudou, stream=True).raw
image_array_rondoudou = np.asarray(bytearray(resp.read()), dtype="uint8")






image_pikachu = cv2.imdecode(image_array_pikachu, cv2.IMREAD_COLOR)

image_rondoudou = cv2.imdecode(image_array_rondoudou, cv2.IMREAD_COLOR)
#print(cv2.imdecode(image_array_rondoudou, cv2.IMREAD_COLOR).shape)



# Image pikachu mis au format rgb 
plt.axis('off')
plt.imshow(cv2.cvtColor(image_pikachu, cv2.COLOR_BGR2RGB)) #opencv if BGR color, matplotlib usr RGB so we need to switch otherwise the pikachu will be blue ... O:)
plt.show()

# Image pikachu mis au format rgb 
plt.axis('off')
plt.imshow(cv2.cvtColor(image_rondoudou, cv2.COLOR_BGR2RGB))
#print(cv2.cvtColor(image_rondoudou, cv2.COLOR_BGR2RGB).shape)
plt.show()


#♦resize
res = cv2.resize(image_pikachu , dsize=(40,40), interpolation=cv2.INTER_CUBIC)
print(res.shape)

res = cv2.cvtColor(res,cv2.COLOR_RGB2GRAY) #TO 3D to 1D
print(res.shape)
res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)[1]
d = res

plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
#

res1 = cv2.resize(image_rondoudou , dsize=(40,40), interpolation=cv2.INTER_CUBIC)

res1 = cv2.cvtColor(res1,cv2.COLOR_RGB2GRAY) #TO 3D to 1D
res1 = cv2.threshold(res1, 127, 255, cv2.THRESH_BINARY)[1]
d1 = res1

plt.imshow(cv2.cvtColor(res1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# 



img_bw = cv2.imdecode(image_array_pikachu, cv2.IMREAD_GRAYSCALE)
(thresh, img_bw) = cv2.threshold(img_bw, 127, 255, cv2.THRESH_BINARY)
plt.axis('off')
plt.imshow(cv2.cvtColor(img_bw, cv2.COLOR_BGR2RGB))


#defining an identity kernel, will change nothing because each pixel will remain with is value
kernel = np.matrix([[0,0,0],[0,1,0],[0,0,0]])
img_1 = cv2.filter2D(img_bw, -1, kernel)

kernel = np.matrix([[-10,0,10],[-10,0,10],[-10,0,10]])
img_1 = cv2.filter2D(img_bw, -1, kernel)
plt.axis('off')
plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))


#defining an horizontal edge detection kernel 
kernel = np.matrix([[10,10,10],[0,0,0],[-10,-10,-10]])
img_1 = cv2.filter2D(img_bw, -1, kernel)
plt.axis('off')
plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))


#Import dataset

import pathlib
import os
data_dir = tf.keras.utils.get_file(
    "dataset.zip",
    "https://github.com/anisayari/Youtube-apprendre-le-deeplearning-avec-tensorflow/blob/master/%234%20-%20CNN/dataset.zip?raw=true",
    extract=False)

import zipfile
with zipfile.ZipFile(data_dir, 'r') as zip_ref:
    zip_ref.extractall('/content/datasets')

data_dir = pathlib.Path('/content/datasets/dataset')
print(data_dir)
print(os.path.abspath(data_dir))


'''
data_dir = pathlib.Path('dataset')
image_count = len(list(data_dir.glob('*/*')))
print(image_count)

batch_size = 3
img_height = 200
img_width = 200

train_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  )

val_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = val_data.class_names
print(class_names)


plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
  for i in range(3):
    ax = plt.subplot(1, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
    
    
from tensorflow.keras import layers

num_classes = 2

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(128,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'],)

logdir="logs"

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1, write_images=logdir,
                                                   embeddings_data=train_data)

model.fit( 
    train_data,
  validation_data=val_data,
  epochs=4,
  callbacks=[tensorboard_callback]
)

model.summary()

import glob
from PIL import Image


path1 = 'predict'
listing = os.listdir(path1)    
for file in listing:
    im_to_predict = Image.open(path1 +"/"+ file)
    
    image_to_predict1 = cv2.imread(path1 +"/"+ file,cv2.IMREAD_COLOR)
    
    
    plt.imshow(cv2.cvtColor(image_to_predict1, cv2.COLOR_BGR2RGB))
    plt.show()
    img_to_predict1 = np.expand_dims(cv2.resize(image_to_predict1,(200,200)), axis=0) 
    
    
    
    
    
    res3 = model.predict(img_to_predict1)
    print(model.predict(img_to_predict1))
    print(res3[0][0])
    if res3[0][0] > 0.5:
        res3=1
    else:res3=0
    if res3 == 1:
        plt.imshow(cv2.cvtColor(im_to_predict, cv2.COLOR_BGR2RGB))
        plt.title('pika')
        plt.show()
        print("IT'S A PIKACHU !")
    elif res3 == 0 :
        plt.imshow(cv2.cvtColor(im_to_predict, cv2.COLOR_BGR2RGB))
        plt.title('rondoudou')
        plt.show()
        
        print("IT'S A RONDOUDOU !")

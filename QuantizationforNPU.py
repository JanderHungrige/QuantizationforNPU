#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:01:22 2020

@author: base

#--------------------------------
# If the link to the model should fail , it can be loaded and save with the following:
import subprocess
import sys

def install(git+https://github.com/rcmalli/keras-vggface.git):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from keras_vggface.vggface import VGGFace
pretrained_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max
#pretrained_model.summary()
pretrained_model.save("my_model.h5") #using h5 extension
#-----------------------
"""


import tensorflow as tf
import cv2
from keras_vggface.utils import preprocess_input
import numpy as np
from pathlib import Path
print(tf.version.VERSION)
if tf.__version__.startswith('1.15'):
    # This prevents some errors that otherwise occur when converting the model with TF 1.15...
    tf.enable_eager_execution() # Only if TF is version 1.15


path_to_model=Path('my_model.h5')
path_to_img=Path('000002.jpg')
print(tf.version.VERSION)

if path_to_model.is_file():
    if tf.__version__.startswith('2.'):
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(path_to_model) #works now also with TF2.x
    if tf.__version__.startswith('1.'):
        converter = tf.lite.TFLiteConverter.from_keras_model_file(path_to_model)  
else:
    print('Please add the my_model.h5 to the working directory or change the path')
    
def representative_dataset_gen():
    if path_to_img.is_file():
      for _ in range(10):
        img=cv2.imread(path_to_img)
        img = np.expand_dims(img, axis=0).astype('float32')
        img = preprocess_input(img, version=2) 
        yield [img]
    else:
        print('Please add the example image or a 224x224 image to the working directory or change the path')
    
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.experimental_new_converter = True

converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8 
converter.inference_output_type = tf.int8 
quantized_tflite_model = converter.convert()
if tf.__version__.startswith('1.'):
    open("tf1_15_3_all_int8.tflite", "wb").write(quantized_tflite_model)
if tf.__version__.startswith('2.'):
    open("tf220_all_int8.tflite", 'wb') .write(quantized_tflite_model)# mit 220 vs 2_2_0 Ich hatte im modelcode dtype int32 und int8 eingefüght. Jetzt wieder draußen
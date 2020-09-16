#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:01:22 2020

@author: base


The model can be loaded with 

pip install git+https://github.com/rcmalli/keras-vggface.git


"""


import tensorflow as tf
import cv2
from keras_vggface.utils import preprocess_input
import numpy as np

print(tf.version.VERSION)
if tf.__version__.startswith('1.15'):
    # This prevents some errors that otherwise occur when converting the model with TF 1.15...
    tf.enable_eager_execution() # Only if TF is version 1.15

#--------------------------------
# If I fail to send you the model, it can be loaded and save with the following
from keras_vggface.vggface import VGGFace

pretrained_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max
#pretrained_model.summary()
pretrained_model.save("my_model.h5") #using h5 extension

#-----------------------

fullint=True
saved_model_dir='PATH_where you saved the model/'
modelfile='my_model.h5'

Datentyp='int8'   #'int8' or 'uint8'

print(tf.version.VERSION)

if tf.__version__.startswith('2.'):
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(saved_model_dir + modelfile) #works now also with TF2.x
if tf.__version__.startswith('1.'):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_model_dir + modelfile)  
    
def representative_dataset_gen():
  for _ in range(10):
    pfad='Path to a sample image'
    img=cv2.imread(pfad)
    img = np.expand_dims(img, axis=0).astype('float32')
    img = preprocess_input(img, version=2) 
    yield [img]
    
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
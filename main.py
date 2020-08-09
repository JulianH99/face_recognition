#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 11:22:28 2020

@author: julian
"""



import tensorflow as tf
import keras as khe

import csv

import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical



def create_emotion_folder(emotion):
    emotions_folder = os.path.join(os.getcwd(), 'emotions')
    
    if not os.path.isdir(emotions_folder):
        os.mkdir('emotions')
    
    emotion_path = os.path.join(emotions_folder, emotion)
    
    if not os.path.isdir(emotion_path):
        os.mkdir(emotion_path)



def classify(images_path='data'):
    csv_file_path = os.path.join(os.getcwd(), images_path, 'legend.csv')

    print(csv_file_path)
    
    file = open(csv_file_path, 'r')
    
    csv_reader = csv.reader(file, delimiter=',')
    
    emotion_names = []
    
    for csv_line in csv_reader:
        image_name = csv_line[1]
        emotion = csv_line[2].lower()
        
        emotion_names.append(emotion)
        
        
        image_path = os.path.join(os.getcwd(), 'images', image_name)
        image_exists = os.path.isfile(image_path)
        
        image_emotion_path = os.path.join(os.getcwd(), 
                                          'emotions', 
                                          emotion,
                                          image_name)
        
        create_emotion_folder(emotion)
        
        if image_exists:
            os.rename(image_path, image_emotion_path)
            
        
    return emotion_names


def load_images(test_path):
    
    images_array = []
    
    for image in os.listdir(test_path):
        l_image = load_img(os.path.join(test_path, image))
                
        images_array.append(img_to_array(l_image))
        
        
    return images_array


def use_tf():
    images_train_path = os.path.join(os.getcwd(), 'emotions')
    test_path = os.path.join(os.getcwd(), 'test')
    
    images_train = ImageDataGenerator(rescale=1./255)
    # test_data = ImageDataGenerator(rescale=1./255)
    
    
    # test_images = load_images(test_path)
    
    
    images_generator = images_train.flow_from_directory(images_train_path,
                                                        target_size=(350, 350))
    # test_generator = test_data.flow(test_images)
    
    
    # Design
    # Se define como un modelo secuencial
    model = Sequential()
    
    # Se añaden las capas y sus hiperparámetros
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(350, 350, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #40x40
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #20x20
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #10x10
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #5x5
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    # La capa de salida debe tener el mismo número de clases
    model.add(Dense(9, activation='softmax'))
    model.summary()
    
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    
    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5),
        ModelCheckpoint(filepath=os.path.join(os.getcwd(), 'model_compiled.h5'), 
                        monitor='val_loss', 
                        save_best_only=True, 
                        verbose=1,
                        mode='min')
    ]
    
    print(images_generator.n/images_generator.batch_size)
    model.fit(images_generator, epochs=32,
              steps_per_epoch=images_generator.n/images_generator.batch_size,
              callbacks=callbacks)
    
    
classify()

use_tf()
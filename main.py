#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 11:22:28 2020

@author: julian
"""






import csv

import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2



face_cascade = cv2.CascadeClassifier('./cascade.xml')

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
    test_data = ImageDataGenerator(rescale=1./255)
    
    
    # test_images = load_images(test_path)
    
    
    images_generator = images_train.flow_from_directory(images_train_path,
                                                        target_size=(350, 350))
    test_generator = test_data.flow_from_directory(test_path,
                                                   target_size=(350, 350))
    
    
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
    
    
    
    
    model_path = os.path.join(os.getcwd(), 'model_compiled.h5' )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5),
        ModelCheckpoint(filepath=model_path, 
                        monitor='val_loss', 
                        verbose=1,
                        mode='min')
    ]
    
    print(images_generator.n/images_generator.batch_size)
    
    
    print("Model path", model_path)
    if not os.path.isfile(model_path):
        print("Generating model")
        model.fit(images_generator, epochs=1,
              steps_per_epoch=images_generator.n/images_generator.batch_size,
              callbacks=callbacks)
        
    print("Loading model")
    model_loaded = load_model(model_path)
    
    
    step_size_test=test_generator.n/test_generator.batch_size
    result_evaluate =  model_loaded.evaluate_generator(test_generator,step_size_test,verbose=1)
    
    
    y_pred_prob =  model_loaded.predict_generator(test_generator, steps= step_size_test)
    
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    
    test_labels_one_hot = to_categorical(test_generator.classes)
    
    print(y_pred_prob)
    
    
    videocam(model_loaded)
    
    
    

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,350,350),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face
    


def videocam(model):
    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        #canvas = detect(gray, frame)
        #image, face =face_detector(frame)
        
        face=face_extractor(frame)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (350, 350))
            im = Image.fromarray(face, 'RGB')
               #Resizing into 128x128 because we trained the model with this image size.
            img_array = np.array(im)
                        #Our keras model used a 4D tensor, (images x height x width x channel)
                        #So changing dimension 128x128x3 into 1x128x128x3 
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            print(pred)
                         
            name="None matching"
            
            if(pred[0][3]>0.5):
                name='Krish'
            #cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,350,0), 2)
        else:
            cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,350,0), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    
classify()

use_tf()
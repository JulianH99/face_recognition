# -*- coding: utf-8 -*-


# organize imports
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet, MobileNetV2, ResNet50, VGG16, VGG19

images_train_path = os.path.join(os.getcwd(), 'emotions')
images_train = ImageDataGenerator(rescale=1./255)
images_generator = images_train.flow_from_directory(images_train_path,
                                                        target_size=(350, 350))

mobilenet = MobileNet()
prediction = mobilenet.predict(images_generator)


mobilenetv2 = MobileNetV2()
prediction = mobilenetv2.predict(images_generator)


resnet50 = ResNet50()
prediction = resnet50.predict(images_generator)


vgg16 = VGG16()
prediction = vgg16.predict(images_generator)


vgg19 = VGG19()
prediction = vgg19.predict(images_generator)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 22:32:04 2019

IMAGE CLASSIFICAION using CNN

@author: arunramji
"""

#PART-1
#initialise neural network
from keras.models import Sequential

#package to perfom first layer , which is convolution , using 2d as it is for image , for video it will be 3d
from keras.layers import Convolution2D

#to perform max pooling on convolved layer
from keras.layers import MaxPool2D

#to convert the pool feature map into large feature vector, will be input for ANN
from keras.layers import Flatten 

#to add layeres on ANN
from keras.layers import Dense

#STEP -1
#Initializing CNN
classifier = Sequential()

#add convolution layer
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),strides=(1, 1),input_shape= (64,64,3),activation='relu'))

#filters - Number of feature detecters that we are going to apply in image

#kernel_size - dimension of feature detector

#strides moving thru one unit at a time

#input shape - shape of the input image on which we are going to apply filter thru convolution opeation,
#we will have to covert the image into that shape in image preprocessing before feeding it into convolution
#channell 3 for rgb and 1 for bw , and  dimension of pixels

#activation - function we use to avoid non linearity in image

#STEP -2 

#add pooling
#this step will significantly reduce the size of feature map , and makes it easier for computation

classifier.add(MaxPool2D(pool_size=(2,2)))

#pool_size - factor by which to downscale

#adding second convolutional layer
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))


#STEP -3
#flattern the feature map

classifier.add(Flatten())

#STEP -4 
#hidden layer
classifier.add(Dense(units=128,activation='relu',kernel_initializer='uniform'))

#output layer
classifier.add(Dense(units=1,activation='sigmoid'))


#Compiling the CNN using stochastic gradient descend

classifier.compile(optimizer='adam',loss = 'binary_crossentropy',
                  metrics=['accuracy'])

#loss function should be categorical_crossentrophy if output is more than 2 class

#PART2 - Fitting CNN to image

#copied from keras documentation 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '/Users/arunramji/Downloads/Sourcefiles/CNN_Imageclassification/Convolutional_Neural_Networks/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
    '/Users/arunramji/Downloads/Sourcefiles/CNN_Imageclassification/Convolutional_Neural_Networks/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=250,   #number of input (image)
        epochs=25,
        validation_data=test_set,
        validation_steps=63)          # number of training sample


#PART 3 - Prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(
        '/Users/arunramji/Downloads/Sourcefiles/CNN_Imageclassification/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_2.jpg',
        target_size = (64,64))
test_image = image.img_to_array(test_image) #making it 3D array as input layer is 3D     
test_image = np.expand_dims(test_image,axis=0) #adding bias variable

result = classifier.predict(test_image)

training_set.class_indices  #to check the value of outplut class assigned

if result[0][0] == 1:
    prediction = "Dog"
    print('I am Robot: It looks like a Dog!')
else:
    prediction = "Cat"
    print('I am Robot: oooh, It looks like a Cat to me')
    












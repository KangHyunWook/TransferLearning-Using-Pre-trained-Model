"""
Date: 2021-Aug-14
Programmer: HYUN WOOK KANG
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Activation, Dense

import numpy as np
import cv2
import os

def preprocess(img_path):
    img=image.load_img(img_path, target_size=(224,224,3))
    img=image.img_to_array(img)
    img=preprocess_input(img)

    return img

def to_categorical(train_y, labels):
    n=len(train_y)
    print('n:',n)
    y=np.zeros((n,2), dtype=np.uint8)
    for i in range(len(y)):
        index=labels.index(train_y[i])
        y[i,index]=1        
    return y
"""Prepare train data"""
labels=['dog','cat']

data_path='./data'

categories=os.listdir(data_path)
train_X=[]
train_y=[]
for i in range(len(categories)):
    files=os.listdir(os.path.join(data_path,categories[i])) 
    for j in range(len(files)):
        img_path=os.path.join(data_path,categories[i],files[j])
        img=preprocess(img_path)
        label=img_path.split('\\')[-2]        
        train_X.append(img)
        train_y.append(label)

train_X=np.array(train_X)
train_y=to_categorical(train_y,labels)

print(train_X.shape)
print(train_y.shape)
            
"""Declare a Larger network using pre-trained model, vgg16 with the full connected layers removed"""
vgg16=VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
# features=vgg16.predict(img)

"""create small network composed of two hidden layers with 300 nodes"""
x = Flatten()(vgg16.output)
x = Dense(300)(x)
Activation('relu')(x)
x=Dense(300)(x)
Activation('relu')(x)
x=Dense(2)(x)
x=Activation('sigmoid')(x)

"""An enhanced model is created by adding vgg16 to small network"""
model=Model(inputs=vgg16.input, outputs=x)

for layer in vgg16.layers:
    layer.trainable=False
print(x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(train_X, train_y, epochs=1, batch_size=4)

"""Just to remind you that images with below file names are not included in the training set"""
test_img_file_names=os.listdir('./testimgs')
print('test image file name\tpredictions')
for i in range(len(test_img_file_names)):
    test_img_path=os.path.join('./testimgs',test_img_file_names[i])
    test_img=preprocess(test_img_path)
    test_img=np.expand_dims(test_img,axis=0)
    probs=model.predict(test_img)
    # print(probs)
    index=np.argmax(probs[0])
    print('{}\t\t{}'.format(test_img_file_names[i],labels[index]))








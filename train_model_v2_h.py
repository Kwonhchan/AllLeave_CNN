import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import layers
from keras.utils import plot_model
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
import os
import matplotlib.pyplot as plt
import preprocessing as pp
import h5py
from keras.models import load_model


# CNN_2
def create_model_2():
    model_2=keras.Sequential()

    model_2.add(layers.Conv2D(64, (15,15), activation='relu', input_shape=(256,256,3), padding='same', strides=(3,3)))
    model_2.add(BatchNormalization())
    model_2.add(layers.MaxPooling2D(pool_size=(8,8)))

    model_2.add(layers.Conv2D(32, (3,3), activation='relu', padding='same',strides=(3,3)))
    model_2.add(BatchNormalization())
    model_2.add(layers.MaxPooling2D(pool_size=(2,2)))

    # DNN_2
    model_2.add(layers.Flatten())
    model_2.add(layers.Dropout(0.3))
    model_2.add(layers.Dense(128, activation='relu'))
    model_2.add(layers.Dense(10,activation='softmax'))

    return model_2

#데이터 나누기
def split():
    images,labels=pp.processing()
    trainX, testX, trainY, testY = train_test_split(images, labels, test_size=0.1, random_state=25, stratify = labels)
    trainX = trainX / 255
    testX = testX / 255 

    return trainX,testX,trainY,testY

#모델 훈련
def train():
    
    #모델 불러오기
    model = create_model_2()

    #데이터 분리 / stratify를 사용하는 이유는 현재 두 개의 데이터셋의 개수가 균일하지 않기 때문 > 빅벤 100 언저리, 산토리니 400언저리
    trainX, testX, trainY, testY = split()

    #이미지 파일 > 정규화
    trainX = trainX / 255
    testX = testX / 255 

    #모델 컴파일 [sparse_categorical_crossentropy] : one-hot encoding 없이 정수 인코딩 된 상태에서 사용
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist1 = model.fit(trainX, trainY, epochs=30, batch_size=20)

    #모델 평가
    model.evaluate(testX, testY)

    plt.subplot(1,2,1)
    plt.plot(hist1.history['loss'])
    plt.subplot(1,2,2)
    plt.plot(hist1.history['accuracy'])
    plt.show()

    #모델 저장
    model_filename = 'train_model_v1.h5'
    model.save(model_filename)



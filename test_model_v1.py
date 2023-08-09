from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import train_model_v1 as v1
import tensorflow as tf
import tensorflow_hub as hub
import os
from PIL import Image
import random 

def test()->None:
    categories = {
            0 : 'Bigben' ,
            1 : 'Santorini',
            2 : 'Matterhorn',
            3 : 'Grand_Canyon',
            4 : 'the_statue_of_liberty',
            5 : 'eiffel_tower' ,
            6 : 'Gold_gate_bridge', 
            7 : 'Osakajo', 
            8 : 'pisa_tower', 
            9 : 'ayasofya_camii' 

        }
    #모델 이름 입력
    Mname = 'BigTransferModel.h5'

    #모델 불러오기
    with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
        loaded_model = tf.keras.models.load_model(Mname)

    #테스트 이미지 경로
    path_t = r'data_set\test_images'

    img_array = []
    img_files = [os.path.join(path_t, filename) for filename in os.listdir(path_t)]
    for img_file in img_files:
        try:
            img = Image.open(img_file)
            img = np.array(img)
            print(img.shape)
        
            img = tf.image.resize(img, [256, 256]) / 255.0
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            img = np.expand_dims(img, axis=0)
            img_array.append(np.array(img))
            
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")


    if img_array:
        data_file = np.concatenate(img_array, axis=0)
        print(data_file.shape)
        pred = loaded_model.predict(data_file)
    
    pred_labels = np.argmax(pred, axis=1)

    plt.figure(figsize=(10,10))

    num_images = len(img_array)

    for i in range(num_images):
        plt.subplot(10, 10, i+1)
        img = np.array(img_array[i])
        img = img.reshape((256,256,3))
        plt.imshow(img)
        plt.title(categories[pred_labels[i]])
        plt.axis('off')
    plt.show()


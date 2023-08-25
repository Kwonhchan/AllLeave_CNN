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

class img_processing:
        
    categories = [
        'bigben',
        'santorini',
        'Matterhorn',
        'Grand_Canyon',
        'the_statue_of_liberty',
        'eiffel_tower',
        'Gold_gate_bridge',
        'Osakajo',
        'pisa_tower',
        'ayasofya_camii',
        'Donggung_Palace_and_Wolji_Pond',
        'Gobi_Desert',
        'Iceland_Aurora',
        'kuta_beach',
        'Machu_Picchu',
        'Niagara_falls',
        'Pyramid',
        'Salar_de_Uyuni',
        'Sydney_Opera_House',
        'Torre_pendente_di_Pisa',
        'Wat_Chedi_Luang'

    ] # 지역이름 추가하기
        
    def img_label(path,region):


        categories = {
            'bigben' : 0 ,
            'santorini' : 1,
            'Matterhorn' : 2,
            'Grand_Canyon' : 3,
            'the_statue_of_liberty' : 4,
            'eiffel_tower' : 5,
            'Gold_gate_bridge' : 6,
            'Osakajo' : 7,
            'pisa_tower' : 8,
            'ayasofya_camii' : 9,
            'Donggung_Palace_and_Wolji_Pond' : 10,
            'Gobi_Desert' : 11,
            'Iceland_Aurora' : 12,
            'kuta_beach' : 13,
            'Machu_Picchu' : 14,
            'Niagara_falls' : 15,
            'Pyramid' : 16,
            'Salar_de_Uyuni' : 17,
            'Sydney_Opera_House' : 18,
            'Torre_pendente_di_Pisa' : 19,
            'Wat_Chedi_Luang' : 20
        }
        label = []
        images = []

        img_file = os.listdir(path)
        for img in img_file:
            label.append(categories[region])
            image = keras.preprocessing.image.load_img(f'{path}/{img}',target_size=(256,256))
            imageArr = np.array(image)
            images.append(imageArr)
            
        return images,label
            

    def concat(path,region,X,Y):
        NEW_X = np.concatenate((X,np.array(img_processing.img_label(path,region)[0])),axis=0)
        NEW_Y = np.concatenate((Y,np.array(img_processing.img_label(path,region)[1])),axis=0)
        return NEW_X, NEW_Y 

def processing():


    images = np.empty((0,256,256,3)) # 배열 생성
    labels = np.empty(0) # 배열생성

    for cate in img_processing.categories:
        images,labels = img_processing.concat(f'data_set/'+cate,cate,images,labels)
    print(images.shape,labels.shape)
    return images,labels

if __name__ == '__main__':  
    images,labels = processing()
    print(images.shape,labels.shape)

    




import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import basename, join
from scipy.signal import find_peaks
import re
import pandas as pd

def get_position(peaks_posicion, peaks_intensidad, w):
    dict_value = dict(zip(peaks_posicion, peaks_intensidad))
    dict_value = {k: v for k, v in sorted(dict_value.items(), key=lambda item: item[1], reverse=True)}
    d = w
    for i in dict_value:
        if np.abs(i-w)<d:
            d = np.abs(i-w)
            idx = i
    return idx

def get_peak(gray):
    mean_intensidad = np.mean(gray, axis=0)
    peaks, _ = find_peaks(mean_intensidad, height=0)
    _,w = gray.shape
    w = w//2
    idx = get_position(peaks, mean_intensidad[peaks], w)
    return idx 

def create_dataset_digitos(path):
    if 'div' not in os.listdir(path):
        os.mkdir(join(path,'div'))
    for i in os.listdir(path):
        print(i)
        if 'jpg' not in i:
            continue
        img = cv2.imread(join(path,i))
        ### convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ### recortar por columna
        idx = get_peak(gray)    
        izq = img[:,:idx,:]
        der = img[:,idx:,:]
        cv2.imwrite(path+'div/'+i.split('.')[0]+'_izq'+'.jpg', izq)
        cv2.imwrite(path+'div/'+i.split('.')[0]+'_der'+'.jpg', der)

def get_label_digitos(path, df):
    list_dict = []

    for i in os.listdir(path):
        if 'jpg' not in i:
            continue
        img = cv2.imread(join(path,i))
        a = path.split('/')[3]
        b = path.split('/')[-2]
        name = f'../data/images_digitos/{a}_{b}_{i}'
        cv2.imwrite(name, img)
        index = re.search(r'\d+',i)   
        id_img = 'C_'+index.group(0)
        val = df.set_index('id').loc[id_img]['date_'+b]
        try:
            if 'izq' in i:
                list_dict.append({'id':id_img+'_izq', 'label':val[0],'name':basename(name)})
            else:
                list_dict.append({'id':id_img+'_der', 'label':val[1],'name':basename(name)})
        except:
            list_dict.append({'id':id_img+'_none', 'label':10,'name':basename(name)})
            pass
        
    return list_dict

if __name__=='__main__':
    list_path = ['../data/output/image_train_transform/fecha/upscaling/digitos/day/',
                '../data/output/image_train_transform/fecha/upscaling/digitos/month/']
    for path in list_path: 
        create_dataset_digitos(path)

    print('################# Etiquetando Imagenes de Digitos #############################')
    list_path_div = ['../data/output/image_train_transform/fecha/upscaling/digitos/day/div',
                     '../data/output/image_train_transform/fecha/upscaling/digitos/month/div']

    if 'images_digitos' not in os.listdir('../data'):
        os.mkdir('../data/images_digitos')

    df_output_train = pd.read_csv('../data/output_train.csv', dtype='str')
    list_dict_total = []
    for path in list_path_div:
        list_dict_total.extend(get_label_digitos(path, df_output_train))

    label_digitos = pd.DataFrame(list_dict_total)
    if 'label_digitos.csv' in os.listdir('../data/'):
        print('############ Entrando #######################')
        label_digitos_last = pd.read_csv('../data/label_digitos.csv', sep=',', encoding='utf-8')
        label_digitos = pd.concat([label_digitos,label_digitos_last])
        label_digitos.to_csv('../data/label_digitos.csv', sep=',', index=False, encoding='utf-8')

    else:
        label_digitos.to_csv('../data/label_digitos.csv', sep=',', index=False, encoding='utf-8')


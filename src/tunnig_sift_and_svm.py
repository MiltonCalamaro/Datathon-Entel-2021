import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os
from itertools import product

import warnings
warnings.simplefilter("ignore")

def extract_sift(ruta_img, dimension_sift):
    descriptor = cv2.SIFT_create(nfeatures=dimension_sift)
    img = cv2.imread(ruta_img)   
    cv_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp, des = descriptor.detectAndCompute(cv_gray,None)
    return des

def load_dataset(ruta_data, dimension_sift):
    data = []
    for i in tqdm(os.listdir(ruta_data)):
        ruta_img = ruta_data + i
        gray = extract_sift(ruta_img, dimension_sift)
        data.append(gray)        
    return data

def vectorKmeans(array,kmeans,k):
    row, col = array.shape
    vector = np.zeros(k)
    for i in range(row):
        k = kmeans.predict(array[i:i+1,:])[0]
        vector[k]=vector[k]+1
    return vector

def get_bag_of_visual_words(data_train,data_test,kmeans,k):
    sift_kmeans_train = []
    for vector in data_train:
        sift_kmeans_train.append(vectorKmeans(vector,kmeans,k))
    sift_kmeans_test = []
    for vector in data_test:
        sift_kmeans_test.append(vectorKmeans(vector,kmeans,k))

    sift_kmeans_train = np.array(sift_kmeans_train)
    sift_kmeans_test = np.array(sift_kmeans_test)
    return sift_kmeans_train, sift_kmeans_test


from sklearn import svm
from sklearn.metrics import classification_report
from imblearn.over_sampling import SVMSMOTE
pattern = re.compile(r'macro avg.*')
def cross_validation(x_train, y_train, parameters, oversample_svm, point_of_court):
    list_fmean_score_test = []
    list_fmean_score_train = []
    for idx in folds:
        x_t = x_train.loc[idx]
        y_t = y_train.loc[idx]
        index = x_t.index
        
        if oversample_svm:
            oversample = SVMSMOTE()
            x_t, y_t = oversample.fit_resample(x_t, y_t)
            
        x_v = x_train.drop(index)
        y_v = y_train.loc[x_v.index]


        clf = svm.SVC(probability=True, random_state = 42, **parameters)
        clf.fit(x_t, y_t)
        pred = clf.predict_proba(x_v)[:,-1]
        pred[pred>=point_of_court] = 1
        pred[pred<point_of_court] = 0
        report = classification_report(y_v, pred)
        fmean_score_test = pattern.search(report).group(0).split()[-2]

        pred = clf.predict_proba(x_t)[:,-1]
        pred[pred>=point_of_court] = 1
        pred[pred<point_of_court] = 0
        report = classification_report(y_t, pred)
        fmean_score_train = pattern.search(report).group(0).split()[-2]

        list_fmean_score_test.append(float(fmean_score_test))
        list_fmean_score_train.append(float(fmean_score_train))

    fmean_score_test_cv = round(np.mean(list_fmean_score_test), 5)
    fmean_score_train_cv = round(np.mean(list_fmean_score_train), 5)
    fmean_score_var = round(np.std(list_fmean_score_test), 5)
    
    return fmean_score_test_cv, fmean_score_train_cv, fmean_score_var

def tunnig_models(x_train, y_train, list_of_parameters, oversample_svm=False, point_of_court=0.5):
    dict_score = {}
    for p in list_of_parameters:
        fmean_score_test_cv, fmean_score_train_cv, fmean_score_var = cross_validation(x_train, y_train, p, 
                                                                                      oversample_svm = oversample_svm, 
                                                                                      point_of_court = point_of_court)
        #print(f'{p} - test: {fmean_score_test_cv} - train: {fmean_score_train_cv} - var: {fmean_score_var} ')
        dict_score[fmean_score_test_cv] = {'parameters':p,'varianza':fmean_score_var,'score_train':fmean_score_train_cv}

    score_test = sorted(dict_score, reverse = True)[0]
    result = {'score_test':score_test, 
              'score_train':dict_score[score_test]['score_train'],
              'varianza':dict_score[score_test]['varianza'],
              'best_parameters':dict_score[score_test]['parameters']}
    return result

def load_ytrain(ruta_data_train, ruta_data_test, column):
    ids = [ i.split('.')[0] for i in os.listdir(ruta_data_train)]
    y_train = pd.read_csv(ruta_label_train,keep_default_na=False, encoding = 'utf-8', dtype = 'str')
    y_train = y_train.set_index('id').loc[ids]
    y_train[column] = y_train[column].astype(int)
    y_train = y_train[[column]].copy()
    return y_train

ruta_label_train = '../data/output_train.csv'
ruta_submit = '../data/sampleSubmission.csv'

def calculate_score(k, dimension_sift, point_of_court, oversample_svm):
    global folds
    #k = oversample_svm
    #dimension_sift= 32
    #point_of_court=0.8
    #oversample_svm=False
    dict_tunning_list = []
    dict_target = {'sign_1':'firma1','sign_2':'firma2','date_day':'fecha','date_month':'fecha','date_year':'fecha'}

    for column in dict_target:
        #column = 'sign_1'
        ruta_data_train = f'../data/output/image_train_out/{dict_target[column]}/'
        ruta_data_test = f'../data/output/image_test_out/{dict_target[column]}/'
        ### cargar y_train
        y_train = load_ytrain(ruta_data_train, ruta_label_train, column)
        if dict_target[column] == 'fecha':
            y_train[column] = y_train[column].apply(lambda x: 1 if x!=0 else 0)

        ### extraccion de caracteristicas
        data_train = load_dataset(ruta_data_train, dimension_sift)
        data_test = load_dataset(ruta_data_test, dimension_sift)

        ### calcular bag of visual words
        from sklearn.cluster import KMeans
        array_final = np.concatenate(data_train,axis = 0)
        kmeans = KMeans(n_clusters=k).fit(array_final)
        sift_kmeans_train, sift_kmeans_test = get_bag_of_visual_words(data_train,data_test,kmeans,k)

        ############### cross_validation and tunning #####################
        from sklearn.model_selection import KFold
        from itertools import product
        x_train = pd.DataFrame(sift_kmeans_train, index = y_train.index)
        folds = [x_train.index[t] for t, v in KFold(5).split(x_train)]

        kernel = ['rbf', 'poly']
        gamma = ['scale',0.1,0.01] 
        C = [ 0.01, 0.1, 1, 10,100]
        parameters = {'kernel':kernel, 'gamma':gamma,'C':C}
        list_of_parameters  = [dict(zip(parameters, v)) for v in product(*parameters.values())]

        dict_tunning = {}
        print(f"{'#'*30} {column} {'#'*30}")
        dict_tunning = tunnig_models(x_train, y_train, list_of_parameters,
                            oversample_svm=oversample_svm, point_of_court=point_of_court)
        dict_tunning['kmeans'] = k
        dict_tunning['oversample_svm'] = oversample_svm
        dict_tunning['dimension_sift']=dimension_sift
        dict_tunning['point_of_court']= point_of_court
        dict_tunning['column'] = column
        print(dict_tunning)
        dict_tunning_list.append(dict_tunning)
    return dict_tunning_list

def main(args):
    kmeans_list = args.kmeans.strip().split()
    kmeans_list = [ int(i) if re.search(r'\d',i) else i for i in kmeans_list]

    dimension_sift_list = args.dimension_sift.strip().split()
    dimension_sift_list = [ int(i) if re.search(r'\d',i) else i for i in dimension_sift_list]

    point_of_court_list = args.point_of_court.strip().split()
    point_of_court_list = [ float(i) if re.search(r'\d',i) else i for i in point_of_court_list]

    if args.oversample_svm:
        oversample_svm_list = [True, False]
    else:
        oversample_svm_list = [False]

    parameters = {'kmeans':kmeans_list, 'dimension_sift':dimension_sift_list, 
                'point_of_court':point_of_court_list, 'oversample_svm':oversample_svm_list}
    list_of_parameters  = [dict(zip(parameters, v)) for v in product(*parameters.values())]

    dict_tunning_list_total = []
    for p in list_of_parameters:
        dict_tunning_list = calculate_score(p['kmeans'], p['dimension_sift'],
                        p['point_of_court'], p['oversample_svm'])
        dict_tunning_list_total.extend(dict_tunning_list)
        pd.DataFrame(dict_tunning_list_total).to_csv('result_score.csv', index = False)
    print(pd.DataFrame(dict_tunning_list_total))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmeans','-k',
                        dest = 'kmeans',
                        help = 'indicar el valor')    
    parser.add_argument('--dimension_sift','-d',
                        dest = 'dimension_sift',
                        help = 'indicar el valor')    
    parser.add_argument('--point_of_court','-p',
                        dest = 'point_of_court',
                        help = 'indicar el valor')    
    parser.add_argument('--oversample_svm','-o',
                        dest = 'oversample_svm',
                        help = 'indicar el valor',
                        action = 'store_true')    
    args = parser.parse_args()
    main(args)
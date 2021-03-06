# coding:utf-8
'''
    @time:    Created on  2018-09-10 20:06:59
    @author:  Lanqing
    @Func:    Define Useful Functions Used in All the Project
'''

import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from config import model_folder, use_pca, saved_dimension_after_pca, \
    test_ratio, whether_shuffle_train_and_test, sample_rate, use_fft, use_gauss, n_estimators, rawFile_read_lines, use_feature_type

def divide_files_by_name(folder_name, different_category):
    '''
        Read files of different categories and divide into several parts
    '''
    import os
    dict_file = dict(zip(different_category, [[]] * len(different_category)))  # initial
    for category in different_category:
        dict_file[category] = []  # Essential here
        for (root, _, files) in os.walk(folder_name):  # List all file names
            for filename in files:  # Attention here
                file_ = os.path.join(root, filename)
                if category in filename:
                    dict_file[category].append(file_)
    return dict_file

def read_single_txt_file_new(single_file_name):
    '''
        Just Load data and Re-sample
    '''   
    import pandas as pd
    # read_array = np.loadtxt(single_file_name, skiprows=20000) 
    df = pd.read_csv(single_file_name, sep=',', header=None)
    read_array = df.values[:, :-1]
    read_array = read_array[:rawFile_read_lines]
    r, c = read_array.shape
    resample_loc = range(int(r / sample_rate))  # Resample
    read_array = read_array[resample_loc, :]
    print('Sample %d * %d, Sample after %d * %d' % (r, c, read_array.shape[0], read_array.shape[1]))
    return read_array

def fft_transform(vector):
    '''
        FFT transform if necessary, only save the real part
        @overwrite: Must Return Ampetitude
    '''
    if use_fft:
        transformed = np.fft.fft(vector)  # FFT
        transformed = transformed.reshape([transformed.shape[0] * transformed.shape[1], 1])  # reshape 
        return np.abs(transformed)
    else:
        return vector

def gauss_filter(X, sigma):
    '''
        Gaussian transform to filter some noise
    '''
    if use_gauss:
        import scipy.ndimage
        gaussian_X = scipy.ndimage.filters.gaussian_filter1d(X, sigma)
        return gaussian_X
    else:
        return X
      
def numericalFeather(singleColumn):
    '''
        Capture the statistic features of a window
    '''
    singleColumn = np.array(singleColumn) 
    medianL = np.median(singleColumn) 
    varL = np.var(singleColumn) 
    meanL = np.mean(singleColumn) 
    static = [medianL, varL, meanL]
    return np.array(static)

def label_encoder(data, train_test_flag):
    '''
        Transform label like 'surfing'/'music' to number between 0-classes-1
        The One-Hot model will be saved to local '.m' file 
    '''
    from sklearn.preprocessing import  LabelEncoder 
    enc = LabelEncoder()
    out = enc.fit_transform(data)  
    if train_test_flag == 'train':
        joblib.dump(enc, model_folder + "Label_Encoder.m")
    return  out, enc

def min_max_scaler(train_data):
    '''
        Min-Max scaler using SkLearn library
        The Model will be saved to local '.m' file
    '''
    from sklearn import preprocessing
    XX = preprocessing.MinMaxScaler().fit(train_data)  
    train_data = XX.transform(train_data) 
    joblib.dump(XX, model_folder + "Min_Max.m")
    return train_data

def vstack_list(tmp):
    '''
        Concentrate a list of array with same number of columns to new array
    '''
    if len(tmp) > 1:
        data = np.vstack((tmp[0], tmp[1]))
        for i in range(2, len(tmp)):
            data = np.vstack((data, tmp[i]))
    else:
        data = np.array(tmp)  
    return data

def hstack_list(tmp):
    '''
        Concentrate a list of array with same number of columns to new array
        @overwrite Deal with Unbalanced problem
    '''
    length_list, re_sized_array = [], []
    for item in tmp:
        length_list.append(len(item))  # calcu all item length
    if len(tmp[0]) * len(tmp) != np.sum(length_list):
        min_length = np.min(length_list)
        for tt in tmp:
            re_sized_array.append(tt[-min_length:]) 
        tmp = np.array(re_sized_array)
    if len(tmp) > 1:
        data = np.hstack((tmp[0], tmp[1]))
        for i in range(2, len(tmp)):
            data = np.hstack((data, tmp[i]))
    else:
        data = np.array(tmp)  
    return data

def validatePR(prediction_y_list, actual_y_list):
    ''' 
        Calculate evaluation metrics according to true and predict labels
    '''
    right_num_dict = {}
    prediction_num_dict = {}
    actual_num_dict = {}
    Precise = {}
    Recall = {}
    F1Score = {}
    if len(prediction_y_list) != len(actual_y_list):
        raise(ValueError)    
    for (p_y, a_y) in zip(prediction_y_list, actual_y_list):
        if p_y not in prediction_num_dict:
            prediction_num_dict[p_y] = 0
        prediction_num_dict[p_y] += 1
        if a_y not in actual_num_dict:  # here mainly for plot 
            actual_num_dict[a_y] = 0
        actual_num_dict[a_y] += 1
        if p_y == a_y:  # basis operation,to calculate P,R,F1
            if p_y not in right_num_dict:
                right_num_dict[p_y] = 0
            right_num_dict[p_y] += 1
    for i in  np.sort(list(actual_num_dict.keys()))  : 
        count_Pi = 0  # range from a to b,not 'set(list)',because we hope i is sorted 
        count_Py = 0
        count_Ri = 0
        count_Ry = 0
        for (p_y, a_y) in zip(prediction_y_list, actual_y_list):
            if p_y == i:
                count_Pi += 1
                if p_y == a_y:                              
                    count_Py += 1  
            if a_y == i :
                count_Ri += 1   
                if a_y == p_y:
                    count_Ry += 1    
        Precise[i] = count_Py / count_Pi if count_Pi else 0               
        Recall[i] = count_Ry / count_Ri if count_Ri else 0
        F1Score[i] = 2 * Precise[i] * Recall[i] / (Precise[i] + Recall[i]) if Precise[i] + Recall[i] else 0
    Micro_average = np.mean(list(F1Score.values()))
    lenL = len(prediction_y_list)
    sumL = np.sum(list(right_num_dict.values()))
    accuracy_all = sumL / lenL
    return Precise, Recall, F1Score, Micro_average, accuracy_all

def check_model():
    '''
        Load model and print the corresponding meaning of the predicted models
    '''
    label_encoder = joblib.load(model_folder + "Label_Encoder.m")
    # print('\n输出类别对应关系：\t', label_encoder.classes_, '\n')
    return label_encoder

def knn_classifier(trainX, trainY, train_wt):  
    '''
        KNN Classifier
    '''
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(trainX, trainY)
    return model

def random_forest_classifier(trainX, trainY, train_wt):  
    '''
        Random Forest Classifier
    '''
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(trainX, trainY, train_wt)
    return model

def PCA(X):
    '''
        Reducing Dimensions using PCA
        The model will be saved to local '.m' file
    '''
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=saved_dimension_after_pca)
        X = pca.fit_transform(X)
        joblib.dump(pca, model_folder + "PCA.m")
    return X

def feature_logic(fft_data, num_feature, train_test_flag):    
    ''' 
        Support different feature method.
        For now the most effective method is using FFT after PCA.
    '''
    if 'PCA' not in use_feature_type:
        if use_feature_type == 'Only_Time_Field':
            data = num_feature
        elif use_feature_type == 'Only_Frequency_Field':
            data = fft_data
        elif use_feature_type == 'Time+Frequency':
            data = np.hstack((num_feature, fft_data))
    elif 'PCA' in use_feature_type:
        if 'train' in train_test_flag:
            if use_feature_type == 'Only_PCA_Frequency':
                fft_data = PCA(fft_data)
                data = fft_data
            elif use_feature_type == 'Time+PCA_Frequency':
                fft_data = PCA(fft_data)
                data = np.hstack((num_feature, fft_data))
        elif 'predict' in train_test_flag:
            pca = joblib.load(model_folder + "PCA.m")
            fft_data = pca.transform(fft_data.T)
            if use_feature_type == 'Only_PCA_Frequency':
                data = fft_data
            elif use_feature_type == 'Time+PCA_Frequency':
                data = np.hstack((num_feature.T, fft_data))
    return data

def generate_configs(train_keyword, train_keyword_, base_folder):
    '''
        Re-Generate config when having to change the processing folder 
    '''
    import os
    base = base_folder 
    print('The folder you want to process is:\t', train_keyword_)
    train_keyword___ = train_keyword[train_keyword_]
    print('The train_keyword are:\t', train_keyword___)
    train_folder = test_folder = predict_folder = base + '/input/' + '/' + train_keyword_ + '/'
    base = base + '/tmp/' + train_keyword_ + '/'
    train_tmp, test_tmp, predict_tmp = base + '/tmp/train/', base + '/tmp/test/', base + '/tmp/predict/' 
    train_tmp_test = base + '/tmp/train/test/'
    model_folder = base + '/model/'
    if not os.path.exists(base):
        os.makedirs(base)
    if not os.path.exists(train_tmp):
        os.makedirs(train_tmp)
    if not os.path.exists(test_tmp):
        os.makedirs(test_tmp)
    if not os.path.exists(predict_tmp):
        os.makedirs(predict_tmp)
    if not os.path.exists(train_tmp_test):
        os.makedirs(train_tmp_test)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    dict_configs = {
        'str1' :"train_keyword = %s" % str(train_keyword___),
        'str2' : "train_folder = '%s'" % str(train_folder),
        'str3' : "test_folder = '%s'" % str(test_folder),
        'str4' : "predict_folder = '%s'" % str(predict_folder),
        'str5' : "train_tmp = '%s'" % str(train_tmp),
        'str6' : "test_tmp = '%s'" % str(test_tmp),
        'str7' : "predict_tmp = '%s'" % str(predict_tmp),
        'str8' : "train_tmp_test = '%s'" % str(train_tmp_test),
        'str9' : "model_folder = '%s'" % str(model_folder),
        'str10': "NB_CLASS = %d" % len(train_keyword___)
    }
    import shutil
    with open('config.py', 'r', encoding='utf-8') as f:
        with open('config.py.new', 'w', encoding='utf-8') as g:
            for line in f.readlines():
                if "train_keyword" not in line and "train_folder =" not in line and "test_folder" not in line \
                    and "predict_folder" not in line and "train_tmp" not in line and "test_tmp" not in line \
                    and "predict_tmp" not in line and "train_tmp_test" not in line and "model_folder =" not in line \
                    and "NB_CLASS" not in line:             
                    g.write(line)
    shutil.move('config.py.new', 'config.py')
    fid = open('config.py', 'a')
    # fid.write('\n')
    for i in range(10):
        fid.write(dict_configs['str' + str(i + 1)])
        fid.write('\n')
    return train_keyword___, train_folder, test_folder, predict_folder, \
        train_tmp, test_tmp, predict_tmp, train_tmp_test, model_folder, len(train_keyword_)    

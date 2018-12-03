# coding:utf-8
'''
@time:    Created on  2018-04-13 18:18:44
@author:  Lanqing
@Func:    Read data and Preprocess
'''
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from config import  sigma, overlap_window, window_length, \
    model_folder, train_data_rate, train_folders, use_feature_type, NB_CLASS
from functions import gauss_filter, fft_transform, divide_files_by_name, read_single_txt_file_new, \
    min_max_scaler, one_hot_coding, PCA, \
    train_test_evalation_split, knn_classifier, random_forest_classifier, validatePR, check_model, generate_configs, \
    vstack_list

TimeStep = 30
epochs = 100
batch_size = 32

def get_LSTM_model(c):
    from keras.layers import LSTM, MaxPooling1D, Conv1D, Reshape
    from keras.layers.core import Dropout, Dense, Activation
    from keras.models import Sequential    
    model = Sequential()
    model.add(LSTM(input_shape=(TimeStep, c), units=128, return_sequences=False))
    # model.add(Dense(64, activation='softmax'))
    model.add(Dense(NB_CLASS, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def LSTM_train(data, label):
    
    from sklearn.model_selection import train_test_split

    #### prepare data
    r, c = data.shape
    new_r = int(r // TimeStep) 
    data = data[:new_r * TimeStep, :].reshape([new_r, TimeStep, c])
    label = label[:new_r * TimeStep].reshape([new_r, TimeStep])    
    label = label[:, -1].reshape([len(label), 1])
    print('Prepared data and label shape:', data.shape, label.shape)
    # print(label)    
        
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.3, shuffle=True)
    model = get_LSTM_model(c)
    # model = get_CNN_LSTM_model(c)
    import keras
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    y_validate = keras.utils.to_categorical(y_validate)    
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validate, y_validate), verbose=1)
    re = model.predict(X_test)
    
    actual_y_list, prediction_y_list = [], []
    for item in y_test:
        actual_y_list.append(np.argmax(item))
    for item in re:
        prediction_y_list.append(np.argmax(item))
    accu = metrics.accuracy_score(actual_y_list, prediction_y_list)
    f2 = metrics.confusion_matrix(actual_y_list, prediction_y_list)
    print('Matrix:\n', f2, '\n accuracy:', accu)
    
    return

def process_a_file(array_, category, after_fft_data):
    '''
        1. Process after "vstack" 
        2. receive an file array and corresponding category
        3. return a clean array
    '''    
    numerical_feature_list, fft_window_list = [], []
    array_ = np.array(array_)
    print(array_.shape)
    if len(array_.shape) > 2:
        _, rows, cols = array_.shape
    else:
        rows, cols = array_.shape
    i = 0
    array_ = array_.reshape([rows * cols, 1])   
    while(i * overlap_window + window_length < rows * cols):  # split window, attention here
        tmp_window = array_[(i * overlap_window) : (i * overlap_window + window_length)]   
        # gauss filter
        tmp_window = gauss_filter(tmp_window, sigma)
        # fft process
        fft_window = fft_transform(tmp_window)
        fft_window_list.append(fft_window)
        numerical_feature_list.append(tmp_window)
        i += 1
    numerical_feature_list = np.array(numerical_feature_list)
    fft_window_list = np.array(fft_window_list)
    fft_array = fft_window_list.reshape([fft_window_list.shape[0], fft_window_list.shape[1]])
    numerical_feature_array = numerical_feature_list.reshape([numerical_feature_list.shape[0], numerical_feature_list.shape[1]])
    return fft_array, numerical_feature_array

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

def data_prepare(input_folder, different_category, percent2read_afterPCA_data, after_fft_data_folder):
    '''
        1. loop all different files , read all into numpy array
        2. label all data
        3. construct train and test
        4. 完成修改，默认读入一个一维的矩阵，降低采样率
    '''
    file_dict = divide_files_by_name(input_folder, different_category)
    fft_list, num_list, label = [], [], []
    for category in different_category:
        files_list = []
        for one_category_single_file in file_dict[category]:  # No worry "Ordered",for it is list
            file_array = read_single_txt_file_new(one_category_single_file)
            files_list.append(file_array)
        file_array_one_category = vstack_list(files_list)
        fft_feature, num_feature = process_a_file(file_array_one_category, category, after_fft_data_folder)  # 预处理
        tmp_label = [category] * len(fft_feature)
        fft_list.append(fft_feature)  # generate label part and merge all
        num_list.append(num_feature) 
        label += tmp_label
    fft_data = vstack_list(fft_list)
    num_feature = vstack_list(num_list)
    data = feature_logic(fft_data, num_feature, 'train')
    label_OneHot, _ = one_hot_coding(label, 'train')  # not really 'one-hot',haha
    data = min_max_scaler(data) 
    print('Shape of data,shape of label:', data.shape, label_OneHot.shape)
    return data, label_OneHot

def baseline_trainTest(data, label):  
    """   
        Train and evaluate using KNN and RF classifier   
    """
    X_train, X_test_left, y_train, y_test_left = train_test_evalation_split(data, label)
    X = X_train
    y = y_train
    print('All samples shape: ', data.shape)
    file_write = model_folder + 'best_model'
    train_wt = None
    # test_classifiers = ['NB','KNN', 'LR', 'RF', 'DT','SVM','GBDT','AdaBoost']
    test_classifiers = ['RF']  # , 'DT','LR']  # , 'GBDT', 'AdaBoost']
    classifiers = {'KNN':knn_classifier,
                   'RF':random_forest_classifier }
    scores_Save = []
    model_dict = {}
    accuracy_all_list = []
    for classifier in test_classifiers:
        scores = []
        skf_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
        i = 0
        for train_index, test_index in skf_cv.split(X, y):
            i += 1                                               
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = classifiers[classifier](X_train, y_train, train_wt)
            predict_y = model.predict(X_test) 
            _, _, _, Micro_average, accuracy_all = validatePR(predict_y, y_test) 
            # print (' \n Precise: %f \n' % Precise, 'Recall: %f \n' % Recall, 'F1Score: %f \n' % F1Score)  # judge model,get score
            scores.append({'cnt:':i, 'mean-F1-Score':Micro_average, 'accuracy-all':accuracy_all})
            accuracy_all_list.append(accuracy_all)
        Micro_average, accuracyScore = [], []
        for item in scores:
            Micro_average.append(item['mean-F1-Score'])
            accuracyScore.append(item['accuracy-all'])
        Micro_average = np.mean(Micro_average)
        accuracyScore = np.mean(accuracyScore)
        scoresTmp = [accuracy_all, Micro_average]
        scores_Save.append(scoresTmp)
        model_dict[classifier] = model 
    scores_Save = np.array(scores_Save)
    print(scores_Save)
    max_score = np.max(scores_Save[:, 1])
    index = np.where(scores_Save == np.max(scores_Save[:, 1]))
    index_model = index[0][0]
    model_name = test_classifiers[index_model]
    print('Test accuracy: ', max_score)
    joblib.dump(model_dict[model_name], file_write)
    model_sort = []
    scores_Save1 = scores_Save * (-1)  ######## 重新调整，打印混淆矩阵
    sort_Score1 = np.sort(scores_Save1[:, 1])  # inverse order
    for item  in sort_Score1:
        index = np.where(scores_Save1 == item)
        index = index[0][0] 
        model_sort.append(test_classifiers[index])
    model = model_dict[model_name]  #### 使用全部数据，使用保存的，模型进行实验    
    predict_y_left = model.predict(X_test_left)  # now do the final test
    _, _, _, Micro_average, accuracy_all = validatePR(predict_y_left, y_test_left) 
    # print ('\n final test: model: %s, F1-mean: %f,accuracy: %f' % (model_sort[0], Micro_average, accuracy_all))
    _ = metrics.accuracy_score(y_test_left, predict_y_left)
    f2 = metrics.confusion_matrix(y_test_left, predict_y_left)
    np.savetxt(model_folder + 'traditional_train_test_confusion_matrix.csv', f2.astype(int), delimiter=',', fmt='%d')
    # f1 = metrics.fbeta_score(y_test_left, predict_y_left,beta= 0.5)
    # print ('Not mine: final test: model: %s,\n accuracy: %f' % (model_sort[0], s1),)
    print('Matrix:\n', f2.astype(int))
    return  accuracy_all_list, max_score

def train(train_folder_defineByUser):
    '''
        Complete code for processing one folder
    '''
    train_keyword, train_folder, _, _, train_tmp, _, _, \
    _, model_folder, _ = generate_configs(train_folders, train_folder_defineByUser)
    # 第二段程序 需要读用户传的命令是什么（训练、测试、预测、基线、模型）
    print(train_keyword)
    data, label = data_prepare(train_folder, train_keyword, train_data_rate, train_tmp)  #### 读数据
    # baseline_trainTest(data, label)  #### 训练KNN、RF等传统模型
    LSTM_train(data, label)
    
def train_test(train_folder_defineByUser):
    train_keyword, train_folder, _, _, train_tmp, _, _, \
    _, model_folder, _ = generate_configs(train_folders, train_folder_defineByUser)
    # 第二段程序 需要读用户传的命令是什么（训练、测试、预测、基线、模型）
    print(train_keyword)
    data, label = data_prepare(train_folder, train_keyword, train_data_rate, train_tmp)  #### 读数据
    # baseline_trainTest(data, label)  #### 训练KNN、RF等传统模型
    LSTM_train(data, label)
    return
    
if __name__ == '__main__':
    train('letter')

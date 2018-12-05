# coding:utf-8
'''
@time:    Created on  2018-04-13 18:18:44
@author:  Lanqing
@Func:    Read data and Preprocess
'''
import numpy as np
import keras as K
from sklearn import metrics
from keras.utils import to_categorical
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from config import sigma, overlap_window, window_length, model_folder, train_folders, NB_CLASS, base_folder, \
     test_ratio, evaluation_ratio, whether_shuffle_train_and_test
from functions import gauss_filter, fft_transform, divide_files_by_name, read_single_txt_file_new, \
    min_max_scaler, label_encoder, feature_logic, knn_classifier, \
    random_forest_classifier, validatePR, check_model, generate_configs, vstack_list

# 控制训练时间
TimeStep = 30
LSTM_unit = 256
epochs, batch_size = 200, 128
callbacks_list = [
    K.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30),  # 30 epoch no improvement
    K.callbacks.ModelCheckpoint(model_folder + 'LSTM_Model.df5',
     monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1) 
    ]

# LSTM Model
def get_LSTM_model(c):
    from keras.layers import LSTM
    from keras.layers.core import Dense, Activation
    from keras.models import Sequential    
    model = Sequential()
    model.add(LSTM(input_shape=(TimeStep, c), units=LSTM_unit, return_sequences=False))
    # model.add(Dense(64, activation='softmax'))
    model.add(Dense(NB_CLASS, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Preprocess
def process_a_file(array_, category, after_fft_data):
    # Reshape
    array_ = np.array(array_)
    if len(array_.shape) > 2:
        _, rows, cols = array_.shape
    else:
        rows, cols = array_.shape
    array_ = array_.reshape([rows * cols, 1])   
    i = 0
    numerical_feature_list, fft_window_list = [], []
    # Moving Window
    while(i * overlap_window + window_length < rows * cols):  
        tmp_window = array_[(i * overlap_window) : (i * overlap_window + window_length)]   
        # Filter
        tmp_window = gauss_filter(tmp_window, sigma)
        # FFT
        fft_window = fft_transform(tmp_window)
        fft_window_list.append(fft_window)
        numerical_feature_list.append(tmp_window)
        i += 1
    # Concat
    numerical_feature_list = np.array(numerical_feature_list)
    fft_window_list = np.array(fft_window_list)
    fft_array = fft_window_list.reshape([fft_window_list.shape[0], fft_window_list.shape[1]])
    numerical_feature_array = numerical_feature_list.reshape([numerical_feature_list.shape[0], numerical_feature_list.shape[1]])
    return fft_array, numerical_feature_array

# Main process
def data_prepare(input_folder, different_category, after_fft_data_folder):
    file_dict = divide_files_by_name(input_folder, different_category)
    train_fft_list, train_num_list, train_label, test_fft_list, test_num_list, test_label = [], [], [], [], [], []
    # Loop All categories
    for category in different_category:
        train_file, test_file = [], []
        # Loop all files per category
        for one_category_single_file in file_dict[category]:
            file_array = read_single_txt_file_new(one_category_single_file)
            if 'train' in one_category_single_file:
                train_file.append(file_array)
            else:
                test_file.append(file_array)
        # Generate
        train_one_category = vstack_list(train_file)
        train_fft_feature, train_num_feature = process_a_file(train_one_category, category, after_fft_data_folder)  
        test_one_category = vstack_list(test_file)
        test_fft_feature, test_num_feature = process_a_file(test_one_category, category, after_fft_data_folder)  
        tmp_train_label = [category] * len(train_fft_feature)
        tmp_test_label = [category] * len(test_fft_feature)
        # Merge
        train_fft_list.append(train_fft_feature)
        test_fft_list.append(test_fft_feature)
        train_num_list.append(train_num_feature)
        test_num_list.append(test_num_feature)
        train_label += tmp_train_label
        test_label += tmp_test_label
    # Merge all categories
    train_fft_data = vstack_list(train_fft_list)
    train_num_feature = vstack_list(train_num_list)
    test_fft_data = vstack_list(test_fft_list)
    test_num_feature = vstack_list(test_num_list)    
    # Preprocess
    train_data = feature_logic(train_fft_data, train_num_feature, 'train')
    test_data = feature_logic(test_fft_data, test_num_feature, 'train')
    train_label_Encoder, _ = label_encoder(train_label, 'train')
    test_label_Encoder, _ = label_encoder(test_label, 'train')
    train_data = min_max_scaler(train_data) 
    test_data = min_max_scaler(test_data) 
    print('Shape of train,shape of test:', train_data.shape, test_data.shape)
    return train_data, test_data, train_label_Encoder, test_label_Encoder

# Baseline: RF
def train_baseline(train_data, test_data, train_label_Encoder, test_label_Encoder):  
    # Initial
    file_write, train_wt = model_folder + 'best_model', None
    test_classifiers = ['RF']
    classifiers = {'KNN':knn_classifier, 'RF':random_forest_classifier }
    scores_Save, model_dict, accuracy_all_list = [], {}, []
    X, y = train_data, train_label_Encoder
    # Using Multiple Classifiers
    for classifier in test_classifiers:
        scores = []
        skf_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        i = 0
        # Cross Validation
        for train_index, test_index in skf_cv.split(X, y):
            i += 1                                               
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = classifiers[classifier](X_train, y_train, train_wt)
            predict_y = model.predict(X_test) 
            _, _, _, Micro_average, accuracy_all = validatePR(predict_y, y_test) 
            scores.append({'cnt:':i, 'mean-F1-Score':Micro_average, 'accuracy-all':accuracy_all})
            print('CV:', {'cnt:':i, 'mean-F1-Score':Micro_average, 'accuracy-all':accuracy_all})
            accuracy_all_list.append(accuracy_all)
        Micro_average, accuracyScore = [], []
        # Calculate Mean
        for item in scores:
            Micro_average.append(item['mean-F1-Score'])
            accuracyScore.append(item['accuracy-all'])
        Micro_average = np.mean(Micro_average)
        accuracyScore = np.mean(accuracyScore)
        scoresTmp = [accuracy_all, Micro_average]
        scores_Save.append(scoresTmp)
        model_dict[classifier] = model 
    # Score
    scores_Save = np.array(scores_Save)
    max_score = np.max(scores_Save[:, 1])
    index = np.where(scores_Save == np.max(scores_Save[:, 1]))
    index_model = index[0][0]
    model_name = test_classifiers[index_model]
    print('Max CV score: %.3f, Best Model: %s ' % (max_score, model_name))
    joblib.dump(model_dict[model_name], file_write)
    model_sort = []
    scores_Save1 = scores_Save * (-1)  ######## 重新调整，打印混淆矩阵
    sort_Score1 = np.sort(scores_Save1[:, 1])  # inverse order
    for item  in sort_Score1:
        index = np.where(scores_Save1 == item)
        index = index[0][0] 
        model_sort.append(test_classifiers[index])
    # Final Test
    model = model_dict[model_name]  #### 使用全部数据，使用保存的，模型进行实验    
    predict_y_left = model.predict(test_data)  # now do the final test
    accuracy = metrics.accuracy_score(test_label_Encoder, predict_y_left)
    f2 = metrics.confusion_matrix(test_label_Encoder, predict_y_left)
    print ('Final test accuracy: %f' % accuracy)
    print('Matrix:\n', f2.astype(int))
    return  accuracy_all_list, max_score

# Train LSTM
def LSTM_train(train_data, test_data, train_label_Encoder, test_label_Encoder):
    # Reshape
    r, c = train_data.shape
    new_r = int(r // TimeStep) 
    train_data = train_data[:new_r * TimeStep, :].reshape([new_r, TimeStep, c])
    train_label = train_label_Encoder[:new_r * TimeStep].reshape([new_r, TimeStep])    
    train_label = train_label[:, -1].reshape([len(train_label), 1])
    # #
    r, c = test_data.shape
    new_r = int(r // TimeStep) 
    test_data = test_data[:new_r * TimeStep, :].reshape([new_r, TimeStep, c])
    test_label = test_label_Encoder[:new_r * TimeStep].reshape([new_r, TimeStep])    
    test_label = test_label[:, -1].reshape([len(test_label), 1])
    print('Prepared data and label shape:', train_data.shape, test_data.shape)
    # Split
    X_train, y_train, X_test, y_test = train_data, train_label, test_data, test_label
    del test_label_Encoder, train_label_Encoder, train_data, train_label, test_data, test_label
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=evaluation_ratio, shuffle=whether_shuffle_train_and_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_validate = to_categorical(y_validate)     
    # Train
    model = get_LSTM_model(c)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, y_test),  # Overwrite
              callbacks=callbacks_list, verbose=1)
    # Predict
    best_model = K.models.load_model(model_folder + 'LSTM_Model.df5')
    re = best_model.predict(X_test)
    # Score
    actual_y_list, prediction_y_list = [], []
    for item in y_test:
        actual_y_list.append(np.argmax(item))
    for item in re:
        prediction_y_list.append(np.argmax(item))
    accu = metrics.accuracy_score(actual_y_list, prediction_y_list)
    f2 = metrics.confusion_matrix(actual_y_list, prediction_y_list)
    print('Confusion Matrix of LSTM:\n', f2, '\n Accuracy:', accu)

# Main
def Main(train_folder_defineByUser):
    # Config
    train_keyword, train_folder, _, _, train_tmp, _, _, _, model_folder, _ = generate_configs(train_folders, train_folder_defineByUser, base_folder)
    # Prepare
    train_data, test_data, train_label_Encoder, test_label_Encoder = data_prepare(train_folder, train_keyword, train_tmp)
    # Train
    # train_baseline(train_data, test_data, train_label_Encoder, test_label_Encoder)  #### 训练KNN、RF等传统模型
    LSTM_train(train_data, test_data, train_label_Encoder, test_label_Encoder)
    # check_model()
    
if __name__ == '__main__':
    Main('letter')

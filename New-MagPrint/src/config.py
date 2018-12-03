# coding:utf-8
'''  
    input and preprocess。
    Train 包括完整的train evaluation test, 
    test 指的是完全相同的数据类型代入计算，
    predict指的是没有标签。 
'''

# 文件处理参数
sample_rate, sensor_sample_rate_now = 1000, 4800  # 采样率, 后者采样率是sensor采样率，不改变硬件不变
window_length = 240  # default 50  # [2, 5, 10, 20, 100]  # 窗口大小
overlap_window = int(0.1 * window_length)  # 窗口和滑动大小
fft_time_step = 0.01
max_entries = int(1000 * 730 / sample_rate)  # 实时绘图窗口大小，单位：绘图窗口总数据包个数，e.g. 200/(4800点/每包730个点) = 30个点
train_data_rate = 1  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)
test_ratio, evaluation_ratio = 0.2, 0.1  # 划分训练、测试、验证集
rawFile_read_lines = 250000  # sometimes the file is not equal length, just for history data plotting or in case imbalanced.

# 预处理
saved_dimension_after_pca, sigma = 0.995 , 5  # 如果算出只取一列，会在PCA函数中被重置
use_gauss, use_pca, use_fft = True, True, True  # True
whether_shuffle_train_and_test = True
use_feature_type = 'Only_PCA_Frequency'

# 模型训练预测
n_estimators = 200  # 随机森林决策树数量

avaliable_feature_type = ['Only_Time_Field', 'Only_Frequency_Field', 'Only_PCA_Frequency', 'Time+Frequency', 'Time+PCA_Frequency']
assert(use_feature_type in avaliable_feature_type)

train_folders = {
            'letter':['a', 'h']
            }

### 此处添加文件相关信息 ###
train_info_file = 'train_info_all.txt'
train_keyword = ['down', 'left', 'right', 'up']
train_folder = '../data//input//lanqing0926/'
test_folder = '../data//input//lanqing0926/'
predict_folder = '../data//input//lanqing0926/'
train_tmp = '../data//tmp/lanqing0926//tmp/train/'
test_tmp = '../data//tmp/lanqing0926//tmp/test/'
predict_tmp = '../data//tmp/lanqing0926//tmp/predict/'
train_tmp_test = '../data//tmp/lanqing0926//tmp/train/test/'
model_folder = '../data//tmp/lanqing0926//model/'
NB_CLASS = 4
# coding:utf-8
'''  配置文件。部分配置由函数生成    '''

# 最主要参数
base_folder = 'E:/DATA/Hack/'  # 全部文件夹主路径
train_folders = { 'letter':['a', 'h'] }  # 分类关键词
sample_rate, sensor_sample_rate_now = 10, 4800  # 数据降维  / 硬件采样率
window_length = 240  # 特征窗口
overlap_window = int(0.1 * window_length)  # 实际划窗比例
rawFile_read_lines = -1  # 每个文件限制行数,-1全读

# 预处理
saved_dimension_after_pca, sigma = 20 , 5  # 如果算出只取一列，会在PCA函数中被重置
use_gauss, use_pca, use_fft = True, True, True  # True
use_feature_type = 'Only_PCA_Frequency'
avaliable_feature_type = ['Only_Time_Field', 'Only_Frequency_Field', 'Only_PCA_Frequency', 'Time+Frequency', 'Time+PCA_Frequency']

# 模型
n_estimators = 200  # 随机森林决策树数量
test_ratio, evaluation_ratio = 0.2, 0.1
whether_shuffle_train_and_test = True

# 自动生成
train_info_file = 'train_info_all.txt'
train_keyword = ['a', 'h']
train_folder = 'E:/DATA/Hack//input//letter/'
test_folder = 'E:/DATA/Hack//input//letter/'
predict_folder = 'E:/DATA/Hack//input//letter/'
train_tmp = 'E:/DATA/Hack//tmp/letter//tmp/train/'
test_tmp = 'E:/DATA/Hack//tmp/letter//tmp/test/'
predict_tmp = 'E:/DATA/Hack//tmp/letter//tmp/predict/'
train_tmp_test = 'E:/DATA/Hack//tmp/letter//tmp/train/test/'
model_folder = 'E:/DATA/Hack//tmp/letter//model/'
NB_CLASS = 2
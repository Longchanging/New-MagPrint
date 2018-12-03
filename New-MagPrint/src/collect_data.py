# coding:utf-8
'''
@time:    Created on  2018-11-30 16:43:06
@author:  Lanqing
@Func:    使用商飞接口，只用来收集和绘制数据
'''

import socket
import numpy as np
from config import window_length, max_entries, sample_rate, fft_time_step
from functions import RealtimePlot
from matplotlib import pyplot as plt
plt.style.use('dark_background')

def new_raw_data_process(content):
    '''
        Process Raw Sensor Data Following Collector Logistics
    '''
    one_package_data = []
    for i in range(6, len(content) - 4):
        if i % 2 == 0:
            prior_byte = content[i]
        else:
            second_byte = content[i]
            num = (prior_byte << 8) | (second_byte)
            if num > 2 ** 15:
                num -= 2 ** 16
            one_package_data.append(num)
        i += 1
    return one_package_data

def reduce_sample_rate(package_list, sample_rate):
    ''' 
        Reduce Sample Rate so that my small PC can process If Necessary
    '''
    package_list = np.array(package_list)
    length_ = len(package_list)
    sample_loc = list(range(0, length_, sample_rate))
    sample_array = package_list[sample_loc]
    return sample_array

def write_one_window_into_file(file_name, D2_list_):
    ''' 
        Write One Window (Window_size * Package_Num) Data Into File 
    '''
    fid = open(file_name, 'a')
    for list_ in D2_list_:
        for item in list_:
            fid.write(str(item))
            fid.write(',')
        fid.write('\n')
    fid.close()
    
def new_real_time(file_folder, file_names, collect_time):
    '''
        Real Time Plotting And Saving Data
    '''
    import time
    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.bind(("0.0.0.0", 8000))
    start_time = time.time()
    sensor_data, pacakage_counter, file_name = {}, {}, {}
    sensor_data['101'], sensor_data['102'] , sensor_data['103'] , sensor_data['104'] = [], [], [], []
    file_name['101'], file_name['102'] , file_name['103'] , file_name['104'] = 'Box_101', 'Arm_102', 'Arm_103', 'Arm_104'
    pacakage_counter['101'], pacakage_counter['102'] , pacakage_counter['103'] , pacakage_counter['104'] = 0, 0, 0, 0
    fig = plt.figure()
    axes1 = fig.add_subplot(111)  # define figures
    plt.xlabel('Time(ms)')
    plt.ylabel('Magnetic Signals(Not uT)')
    display1 = RealtimePlot(axes1)
    Y, X = [], []
    previous_package_time = start_time
    while(True):
        content, destInfo = udpSocket.recvfrom(2048)    
        one_package_data = new_raw_data_process(content)  # get one package data,e.g. 730 data in a package
        print(len(one_package_data))
        address = str(destInfo[0].split('.')[-1])  # sensor address,e.g. 103
        if address and address in sensor_data.keys():  # address in the four sensors
            sensor_data[address].append(one_package_data) 
            time_now, package_data = time.time(), one_package_data  # define x and y axis
            package_data = reduce_sample_rate(package_data, sample_rate)
            time_axis = np.linspace(int((previous_package_time - start_time) * 10000), int((time_now - start_time) * 10000), len(package_data))
            previous_package_time = time_now
            Y.extend(package_data)
            X.extend(time_axis)
            Y = Y[-max_entries:]
            display1.add(time_axis, package_data)
            plt.pause(0.0001)
            pacakage_counter[address] += 1
            if pacakage_counter[address] % window_length == 0:  # every window_size write into file
                write_one_window_into_file(file_folder, sensor_data[address])
                sensor_data[address] = []  # clear and restart to collect window_size of data
            if pacakage_counter[address] % 1000 == 0:
                time_now__ = time.time()
                if (time_now__ - start_time) / 60 > collect_time:  # define collect time, e.g. 5 minutes. Can also manually stop
                    udpSocket.close()  # if manually stop, can set longer collect time.
                    break

##########   History Collect And Real-Time Plot #######
new_real_time('C:/Users/jhh/Desktop/h1.txt', 'a', 3)
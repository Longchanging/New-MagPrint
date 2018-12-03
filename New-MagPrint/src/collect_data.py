# coding:utf-8
'''
@time:    Created on  2018-11-30 16:43:06
@author:  Lanqing
@Func:    使用商飞接口，只用来收集和绘制数据
'''

import socket, time
window_length = 10

# Decode
def Decode(content):
    one_package_data = []
    for i in range(6, len(content) - 4):  # remove unusable data
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

# Write
def write_one_window_into_file(file_name, D2_list_):
    fid = open(file_name, 'a')
    for list_ in D2_list_:
        for item in list_:
            fid.write(str(item))
            fid.write(',')
        fid.write('\n')
    fid.close()
    
# Main
def new_real_time(file_folder, collect_time):
    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.bind(("0.0.0.0", 8000))
    start_time = time.time()
    sensor_data, pacakage_counter = [], 0
    while(True):
        content, _ = udpSocket.recvfrom(2048)    
        one_package_data = Decode(content)  # one package
        sensor_data.append(one_package_data) 
        pacakage_counter += 1
        if pacakage_counter % window_length == 0:  # every 'window' write file
            time_now__ = time.time()
            write_one_window_into_file(file_folder, sensor_data)
            sensor_data = []  # clear restart 
            if (time_now__ - start_time) / 60 > collect_time:  # collect time, minutes
                udpSocket.close()
                break

# #  Collect  ##
new_real_time('C:/Users/jhh/Desktop/null.txt', 3)  # file name,collect minutes
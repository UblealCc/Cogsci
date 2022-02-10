import _pickle as cPickle
import os
import numpy as np
from collections import defaultdict
from numpy.lib.npyio import load
from tqdm import tqdm
import csv

chan = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2',
'P4','P8','PO4','O2']


def data_extract(data_path = " ", load_path = " ", label = 4, vedio = 40, participant = 32, channel = 32, sample = 8064):
    chan = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
    valence = open(load_path + "valence.dat", "w")
    arousal = open(load_path + "arousal.dat", "w")
    valence.close()
    arousal.close()
    for i in tqdm(range(participant)):
        if i < 9:
            dest_file = data_path + "s0" + str(i + 1) + ".dat"
        else:
            dest_file = data_path + "s" + str(i + 1) + ".dat"
    
        data = cPickle.load(open(dest_file, 'rb'), encoding = 'bytes')

        for ve in range(vedio): #40
            feature_file = open(load_path + "features_original.csv", 'w')
            for ch in chan:
                if ch != 'O2':
                    feature_file.write(ch + ",")
                else:
                    feature_file.write(ch)
            feature_file.write("\n")
            for sam in range(sample): #8064
                for ch in range(32): #32
                    if ch == 31:
                        feature_file.write(str(data[b'data'][ve][ch][sam]))
                    else:
                        feature_file.write(str(data[b'data'][ve][ch][sam]) + ",")
                feature_file.write("\n")
            feature_file.close()

            valence = open(load_path + "valence.dat", "a")
            arousal = open(load_path + "arousal.dat", "a")
            if data[b'labels'][ve][0] <= 5:
                valence.write(str(0) + "\n")
            else:
                valence.write(str(1) + "\n")
            
            if data[b'labels'][ve][1] <= 5:
                arousal.write(str(0) + "\n")
            else:
                arousal.write(str(1) + "\n")
            valence.close()
            arousal.close()

            # 对此数据进行处理，主要为减去基准值并存为一个文件
            data_process(load_path = load_path)


def data_process(load_path):
    chan = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
    if not os.path.exists(load_path + "all_data.csv"):
        with open(load_path + "all_data.csv", "w") as f:
            for ch in chan:
                if ch != 'O2':
                    f.write(ch + ",")
                else:
                    f.write(ch)
            f.write("\n")


    columns = np.zeros([8064, 32])
    #读取出值
    with open(load_path + "features_original.csv") as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for num, row in enumerate(reader):# read a row as {column1: value1, column2: value2,...} 共8064组值
            for (k, v) in row.items(): #32个通道的一组值
                columns[num][chan.index(k)] = float(v)
    
    dest_file = open(load_path + "all_data.csv", 'a')
    base_line = np.zeros([128, 32])

    #取到基准值
    for i in range(128):
        base_line[i]= columns[i] + columns[128 + i] + columns[256 + i]
    base_line = base_line / 3


    for i in range(60):
        for j in range(128):
            for ch in chan:
                if ch != 'O2':
                    dest_file.write(str(columns[(i + 3) * 128 + j][chan.index(ch)] - base_line[j][chan.index(ch)]) + ",")
                else:
                    dest_file.write(str(columns[(i + 3) * 128 + j][chan.index(ch)] - base_line[j][chan.index(ch)]))
            dest_file.write("\n")
    dest_file.close()
    
def process_and_get_data(data_path, load_path):
    data_extract(data_path = data_path, load_path = load_path)





from math import trunc
from sklearn.utils import shuffle
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from collections import defaultdict
import os
from scipy.interpolate import griddata

class eeg_dataset2d(Dataset):
    def __init__(self, path = " ", train_seed = 0,save_path = " ", label_name = " ", split = False, target_path = " ",train_data = np.zeros(1), test_data = np.zeros(1), train_label = np.zeros(1), test_label = np.zeros(1), data_type = " "):
        self.path = path
        self.save_path = save_path
        self.data_type = data_type
        self.train_seed = train_seed
        self.arousal = None
        self.valence = None
        if label_name == 'arousal':
            self.arousal = True
        elif label_name == 'valence':
            self.valence = True
        else:
            print("label error !!!")
        self.target_path = target_path
        if split == True:
            self.process_data()
            self.process_label()
            self.final_train_data = np.zeros([10, int(32*40*60* 0.9), 128, 9, 9])
            self.final_test_data = np.zeros([10, int(32*40*60*0.1), 128, 9, 9])
            self.final_train_label = np.zeros([10, int(32*40*60* 0.9), 1])
            self.final_test_label = np.zeros([10, int(32*40*60*0.1), 1])
            self.layer_split()
        if data_type == "train":
            self.train_data = np.zeros([int(32*40*60* 0.9), 128, 9, 9])
            self.train_label = np.zeros([int(32*40*60 * 0.9), 1])
            self.train_data = train_data
            self.train_label = train_label
            self.train_data = self.train_data.astype('float32')
            self.train_label = self.train_label.astype('int64')
        elif data_type == "test":
            self.test_data = np.zeros([int(32*40*60 *0.1), 128, 9, 9])
            self.test_label = np.zeros([int(32*40*60* 0.1), 1])
            self.test_data = test_data
            self.test_label =  test_label
            self.test_data = self.test_data.astype('float32')
            self.test_label = self.test_label.astype('int64')
        elif split != True:
            print("if not split, at least input a dataset type")

    def load_save_data(self):
        print("load and save data")
        des_data = np.zeros([128, 9, 9])
        save_data = np.zeros([128, 32])
        tem_data = np.zeros([1,32])
        chan = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
        with open(self.path + "all_data.csv", "r") as f:
            reader = csv.DictReader(f)
            num = 1
            num_seq = -1
            for seq, row in tqdm(enumerate(reader)):
                for (k, v) in row.items():
                    if k == None:
                        continue
                    tem_data[0][chan.index(k)] = float(v)
                num_seq = num_seq + 1
                save_data[num_seq] = tem_data[0]
                if num_seq == 127:
                    num_seq = -1
                    mean = np.mean(save_data, axis = (0, 1))
                    std = np.std(save_data, axis = (0, 1))
                    new_data = (save_data - mean) / std
                    mean = np.mean(new_data, axis = (0, 1))
                    std = np.std(new_data)
                    assert np.abs(mean) < 1e-15 and np.abs(std - 1) < 1e-10
                    for i in range(128):
                        des_data[i] = self.convert_to_matrix(new_data[i])
                    np.save(self.save_path + "{}.npy".format(num), des_data)
                    num = num + 1
        print("success save data")
    def process_data(self):
        if not os.path.exists(self.save_path + "1.npy"):
            self.load_save_data()
        self.final_data = np.zeros([32, 40*60, 128, 9, 9])
        print("load data")
        for par in range(32):
            for i in range(40 * 60):
                self.final_data[par][i] = np.load(self.save_path + str(par * 40 * 60 + i + 1) + ".npy")
        print("sucessful load data")
    def process_label(self):
        self.final_label = np.zeros([32, 40*60, 1])
        self.tem_label = np.zeros([32, 40, 1])
        label_name = ""
        if self.valence == True:
            label_name = "valence"
        elif self.arousal == True:
            label_name = "arousal"
        par = 0
        ved = 0
        with open(self.path + label_name + ".dat", "r") as f:
            for i, j in enumerate(f):    
                j = j[0]
                for time in range(60):
                    self.final_label[par][ved * 60 + time][0] = int(j)
                self.tem_label[par][ved][0] = int(j)
                if ved == 39:
                    par = par + 1
                    ved = 0
                else:
                    ved = ved + 1       
    def convert_to_matrix(self, dest):

        dest_matrix = np.zeros([9,9])
        dest_matrix[0][3] = dest[0]
        dest_matrix[1][3] = dest[1]
        dest_matrix[2][2] = dest[2]
        dest_matrix[2][0] = dest[3]
        dest_matrix[3][1] = dest[4]
        dest_matrix[3][3] = dest[5]
        dest_matrix[4][2] = dest[6]
        dest_matrix[4][0] = dest[7]
        dest_matrix[5][1] = dest[8]
        dest_matrix[5][3] = dest[9]
        dest_matrix[6][2] = dest[10]
        dest_matrix[6][0] = dest[11]
        dest_matrix[7][3] = dest[12]
        dest_matrix[8][3] = dest[13]
        dest_matrix[8][4] = dest[14]
        dest_matrix[6][4] = dest[15]
        dest_matrix[0][5] = dest[16]
        dest_matrix[1][5] = dest[17]
        dest_matrix[2][4] = dest[18]
        dest_matrix[2][6] = dest[19]
        dest_matrix[2][8] = dest[20]
        dest_matrix[3][7] = dest[21]
        dest_matrix[3][5] = dest[22]
        dest_matrix[4][4] = dest[23]
        dest_matrix[4][6] = dest[24]
        dest_matrix[4][8] = dest[25]
        dest_matrix[5][7] = dest[26]
        dest_matrix[5][5] = dest[27]
        dest_matrix[6][6] = dest[28]
        dest_matrix[6][8] = dest[29]
        dest_matrix[7][5] = dest[30]
        dest_matrix[8][5] = dest[31]
        return dest_matrix
    def return_data(self):
        return self.final_train_data, self.final_test_data, self.final_train_label, self.final_test_label

    def __getitem__(self, index):
        #分别从三维中选取
        dest_data = np.zeros([128, 9, 9])
        dest_label = np.zeros(1)
        if self.data_type == "train":
            dest_data = self.train_data[index]
            dest_label = self.train_label[index]
        elif self.data_type == "test":
            dest_data = self.test_data[index]
            dest_label = self.test_label[index]
        else:
            print("error! must input a dataset type")
        dest_data = dest_data.astype('float32')
        dest_label = dest_label.astype('int64')
        return torch.Tensor(dest_data), torch.Tensor(dest_label) 
    def __len__(self):
        if self.data_type == "train":
            return len(self.train_label)
        elif self.data_type == "test":
            return len(self.test_label)

    def layer_split(self):
        #可尝试改变种子来操控结果
        save_train_index = np.zeros([10, 32 * 40 * 6 * 9])
        save_test_index = np.zeros([10, 32 * 40 * 6 ])
        skf = StratifiedKFold(n_splits=10, random_state = self.train_seed, shuffle=True)
        for par in range(32):
            i = 0
            for train_index, test_index in skf.split(self.final_data[par], self.final_label[par]):
                self.final_train_data[i][(par * 40 * 6 * 9):((par + 1) * 40 * 6 * 9)], self.final_test_data[i][(par * 40 * 6):((par + 1) * 40 * 6)] = self.final_data[par][train_index], self.final_data[par][test_index]
                self.final_train_label[i][(par * 40 * 6 * 9):((par + 1) * 40 * 6 * 9)], self.final_test_label[i][(par * 40 * 6):((par + 1) * 40 * 6)] = self.final_label[par][train_index], self.final_label[par][test_index]
                save_train_index[i][(40 * 6 * 9 * par) : (40 * 6 * 9 * (par + 1))] =par * 40*60 + train_index + 1
                save_test_index[i][(40 * 6 * par) : (40 * 6* (par + 1))] = par * 40*60 +test_index + 1
                
                i = i + 1
        for i in range(10):
            for j in range(32*40*60):
                assert (j + 1 in save_train_index[i]) or (j + 1 in save_test_index[i])
        np.save("root/model/train_test_index/{}/train_index.npy".format('arousal' if self.arousal == True else 'valence'), save_train_index)
        np.save("root/model/train_test_index/{}/test_index.npy".format('arousal' if self.arousal == True else 'valence'), save_test_index)

    
import torch
import numpy as np
from skimage.segmentation import mark_boundaries
import torch.nn.functional as F
from models_2d import model_2d as model_2d
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def normal_data(data):
    max = data.max()
    min = data.min()
    data = (data - min) / (max - min)
    return data

def load_data(path, index):
    load_path = path + str(index) + ".npy"
    return np.load(load_path).astype('float32')

def transform_to_matric(original_data):
    des_data = np.zeros([128,9,9])
    for row in range(128):
        des_data[row] = transform_b(original_data[row])
    return des_data

def transform_to_vector(original_data, train = True):
    des_data = np.zeros([128, 32])
    for row in range(128):
        des_data[row] = transform_a(original_data[row])
    if train:
        ans = np.zeros([3,128,32])
        for i in range(3):
            ans[i] = des_data
        return ans
    else:

        return des_data

def transform_a(dest_matrix):
    dest = np.zeros([32])
    dest[0] = dest_matrix[0][3]
    dest[1] = dest_matrix[1][3]
    dest[2] = dest_matrix[2][2]
    dest[3] = dest_matrix[2][0]
    dest[4] = dest_matrix[3][1]
    dest[5] = dest_matrix[3][3]
    dest[6] = dest_matrix[4][2]
    dest[7] = dest_matrix[4][0] 
    dest[8] = dest_matrix[5][1] 
    dest[9] = dest_matrix[5][3]
    dest[10] = dest_matrix[6][2]
    dest[11] = dest_matrix[6][0]
    dest[12] = dest_matrix[7][3]
    dest[13] = dest_matrix[8][3]
    dest[14] = dest_matrix[8][4]
    dest[15] = dest_matrix[6][4]
    dest[16] = dest_matrix[0][5]
    dest[17] = dest_matrix[1][5]
    dest[18] = dest_matrix[2][4]
    dest[19] = dest_matrix[2][6] 
    dest[20] = dest_matrix[2][8] 
    dest[21] = dest_matrix[3][7] 
    dest[22] = dest_matrix[3][5] 
    dest[23] = dest_matrix[4][4] 
    dest[24] = dest_matrix[4][6] 
    dest[25] = dest_matrix[4][8] 
    dest[26] = dest_matrix[5][7] 
    dest[27] = dest_matrix[5][5] 
    dest[28] = dest_matrix[6][6] 
    dest[29] = dest_matrix[6][8] 
    dest[30] = dest_matrix[7][5] 
    dest[31] = dest_matrix[8][5]
    return dest
def transform_b(dest):
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

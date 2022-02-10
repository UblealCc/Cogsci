from matplotlib import image
import matplotlib.pyplot as plt
from PIL import Image
from numpy import random, true_divide
import torch.nn as nn
import numpy as np
import os, json
import cv2
import torch
from torchvision import  transforms
from torch.autograd import Variable
import torch.nn.functional as F
import sys

from tqdm.std import tqdm
from model_2d import model_2d
import threading



from utiles import load_data,lime_process, get_mask
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
path = "/root/data/DEAP/"

def lime_get(seed,exp_index,model_path,path = path):
    model = model_2d(num_class = 2, load_model = True, load_path = model_path)
    model.eval()
    datas = load_data(path, seed)
    explanation=lime_process(data = datas, exp_index = exp_index)
    mask = get_mask(explanation)
    return mask

def get_original(vedios, tag, exp_index):
    #得到1个视频的原始数据,大小为（32*60， 32）
    des_data = np.zeros([32, 60, 9, 9])
    if tag == 'arousal':
        test_index = np.load("/root/model/train_test_info/arousal/test_index.npy")[2]
        train_index = np.load("/root/model/train_test_info/arousal/arousal/train_index.npy")[2]

        model_paths = "/root/model/load_model/arousal/cross_2_test_acc_91.94010416666667.pkl"
    else:
        test_index = np.load("/root/model/train_test_info/arousal/valence/test_index.npy")[5]
        train_index = np.load("/root/model/train_test_info/arousal/valence/train_index.npy")[5]

        model_paths = "/root/model/load_model/valence/cross_5_test_acc_90.49479166666666.pkl"

    for par in tqdm(range(32)):
        for i in range(60):
            if (par * 40 * 60 + vedios * 60 + i + 1) in test_index:
                des_data[par][i] = lime_get(model_path = model_paths,exp_index = exp_index,seed = par * 40 * 60 + vedios * 60 + i + 1).transpose((2,0,1))[0]
            elif (par * 40 * 60 + vedios * 60 + i + 1) not in train_index:
                print("error")
    np.save("/root/lime/lime_data/epx{}/{}/vedio_{}.npy".format(exp_index,tag,vedios), des_data)
    print("vedio {} down".format(vedios))

if __name__ == '__main__':
    #尝试得到每个视频的统计结果
    for exp in range(1,3):
        get_original(vedios = int(sys.argv[1]), tag = 'arousal', exp_index = exp)
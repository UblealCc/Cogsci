import os
from unittest import result
import numpy as np
from numpy.core.fromnumeric import size
import torch
from torch._C import FloatStorageBase
import torch.nn.functional as F
from torch.nn.modules import conv
import torchvision.models as models
import _pickle as cPickle
import csv
import sys
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import threading


from models_2d import model_2d as model_2d
from models_2d import model_2d_remove as model_2d_rm
from utils_my import load_data, normal_data, transform_a
from gradcam import GradCAM, Guided_backprop
import torchvision.models as modelss

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

def guide_begin(seed, data_path, des_model):
    models = des_model
    get_guide = Guided_backprop(models)
    result = get_guide.visualize(torch.tensor(load_data(data_path, seed).astype('float32')).cuda().unsqueeze(0))
    return result

def gradcam_begin(seed, data_path, des_model):
    models = des_model
    my_model_dict = dict(arch=models, layer_name='relu1', input_size=(9, 9))
    my_gradcam = GradCAM(my_model_dict, True)
    mask, logist= my_gradcam(torch.tensor(load_data(data_path, seed).astype('float32')).cuda().unsqueeze(0))
    return mask

def get_data(type):
    data_paths = "/root/data/DEAP/"
    for label_tag in ['arousal', 'valence']:
        if label_tag == 'arousal':
            train_index = np.load("/root/model/train_test_info/arousal/train_index.npy")[2]
            model_paths = "/root/model/load_model/arousal/cross_2_test_acc_91.94010416666667.pkl"
        else:
            train_index = np.load("/root/model/train_test_info/valence/train_index.npy")[5]
            model_paths = "/root/model/load_model/valence/cross_5_test_acc_90.49479166666666.pkl"
        if type == 'Gradcam':
            models = model_2d(num_class = 2, load_model = True, load_path = model_paths)
        elif type == 'Guided_Gradcam':
            models = model_2d_rm(num_class = 2, load_model = True, load_path = model_paths)
        models.eval()
        models.cuda()
        for par in tqdm(range(32)):
            for ved in range(40):
                for sec in range(60):
                    if (par * 40 * 60 + ved * 60 + sec + 1) in train_index:
                        if type == 'Gradcam':
                            tem = torch.squeeze(gradcam_begin(seed = par *40 * 60 + ved * 60 + sec + 1, data_label = label_tag, des_model = models, data_path = data_paths)).cpu().detach().numpy()
                            np.save("/root/gradcam/data/Gradcam/{}.npy".format(label_tag, par *40 * 60 + ved * 60 + sec + 1), tem)
                        elif type == 'Guided_Gradcam':             
                            tem = np.mean(guide_begin(seed = par *40 * 60 + ved * 60 + sec + 1,des_model = models, data_path = data_paths), axis = 0)
                            gradcam_data = np.load("/root/gradcam/data/Gradcam/{}.npy".format(label_tag, par *40 * 60 + ved * 60 + sec + 1))
                            np.save("/root/gradcam/data/Guided_Gradcam/{}/{}.npy".format(label_tag, par *40 * 60 + ved * 60 + sec + 1), tem * gradcam_data)
                        
def final_gradcam_exp(label_tag, conv_num, type):
    total_data0 = np.zeros([32])
    total_data1 = np.zeros([32])
    total_data = np.zeros([32])
    if label_tag == 'arousal':
        train_index = np.load("/root/model/train_test_info/arousal/train_index.npy")[2]
        model_paths = "/root/model/load_model/arousal/cross_2_test_acc_91.94010416666667.pkl"
    else:
        train_index = np.load("/root/model/train_test_info/valence/train_index.npy")[5]
        model_paths = "/root/model/load_model/valence/cross_5_test_acc_90.49479166666666.pkl"
    model1 = model_2d(num_class = 2, load_model = True, load_path = model_paths)
    model1.eval()
    model1.cuda()
    #加载原始标签
    label = np.load("/root/lime/lime_info/val_{}.npy".format(label_tag))
    for par in tqdm(range(32)):
        for ved in range(40):
            tem_total0 = np.zeros([32])
            tem_total1 = np.zeros([32])
            for sec in range(60):
                if (par * 40 * 60 + ved * 60 + sec + 1) in train_index:
                    if type == 'Gradcam':
                        tem = normal_data(np.load("/root/gradcam/data/Gradcam/{}.npy".format(label_tag, conv_num,par *40 * 60 + ved * 60 + sec + 1)))
                    elif type == 'Guided_Gradcam':
                        tem = normal_data(np.load("/root/gradcam/data/Guided_Gradcam//{}.npy".format(label_tag,conv_num,par *40 * 60 + ved * 60 + sec + 1)))
                    else:
                        print("error")
                    #查看此样本的模型输出标签
                    output= model1(torch.tensor(load_data(path = "/root/data/DEAP/", index = par *40 * 60 + ved * 60 + sec + 1)).cuda().unsqueeze(0))
                    preds = torch.max(output, 1)[1]
                    #提取出gradcam中的32通道
                    tem_data = transform_a(tem)
                    #得到此样本的通道排序
                    order = tem_data.argsort()
                    #若预测标签与真实标签不符，则排除
                    if label[par][ved] <= 5 and preds.cpu().numpy()[0] == 0:
                        j = 32
                        tem_order = np.zeros(32)
                        #对通道票选，当通道值为0时，赋予的票选值不再变化。通道越重要赋予的值小。
                        for i in range(31, -1, -1):
                            if tem_data[order[i]] != 0:
                                tem_total0[order[i]] = j + tem_total0[order[i]]
                                j = j - 1
                            else:
                                tem_total0[order[i]] = tem_total0[order[i]]
                    elif label[par][ved] > 5 and preds.cpu().numpy()[0] == 1:
                        j = 32
                        for i in range(31, -1, -1):
                            if tem_data[order[i]] != 0:
                                tem_total1[order[i]] = j + tem_total1[order[i]]
                                j = j - 1
                            else:
                                tem_total1[order[i]] = tem_total1[order[i]]
                    
            total_data0 = total_data0 + tem_total0
            total_data1 = total_data1 + tem_total1
    label = []
    #得到通道的重要性排序。其中，通道的重要性从高到底对应位置从后到前
    total_data = total_data0 + total_data1
    order_total = total_data.argsort()
    order0 = total_data0.argsort()
    order1 = total_data1.argsort()
    #存储
    if type == 'Gradcam':
        np.save("/root/gradcam/channel_order/Gradcam/conv{}_{}0.npy".format(conv_num, label_tag), order0)
        np.save("/root/gradcam/channel_order/Gradcam/conv{}_{}1.npy".format(conv_num, label_tag), order1)
        np.save("/root/gradcam/channel_order/Gradcam/conv{}_{}.npy".format(conv_num, label_tag), order_total)
    elif type == 'Guide':   
        np.save("/root/gradcam/channel_order/Guided_Gradcam/conv{}_{}0.npy".format(conv_num, label_tag), order0)
        np.save("/root/gradcam/channel_order/Guided_Gradcam/conv{}_{}1.npy".format(conv_num, label_tag), order1)
        np.save("/root/gradcam/channel_order/Guided_Gradcam/conv{}_{}.npy".format(conv_num, label_tag), order_total)
    print("[{}] conv{} down".format(label_tag, conv_num))

if __name__ == '__main__':
    for al_type in ['Gradcam', 'Guided_Gradcam']:
        get_data(type = al_type)
        for label_name in ['arousal', 'valence']:
            for conv_n in range(1, 2):
                final_gradcam_exp(label_tag = label_name, conv_num = conv_n)
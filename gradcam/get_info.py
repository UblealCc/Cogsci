import numpy as np
from models_2d import model_2d
import os
import torch
from tqdm import tqdm
from scipy.interpolate import griddata

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
chan = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']


def mask_channel(cha_name, original_data):
    des_data = np.zeros([128,9,9])
    des_data[cha_name] = original_data[cha_name]
    return des_data.astype('float32')
def get_mask_(chan_name):
    order = {0:(0,3), 1:(1,3), 2:(2,2), 3:(2,0), 4:(3,1), 5:(3,3), 6:(4,2), 7:(4,0), 8:(5,1), 9:(5,3), 10:(6,2), 11:(6,0), 
             12:(7,3), 13:(8,3), 14:(8,4), 15:(6,4),16:(0,5), 17:(1,5), 18:(2,4), 19:(2,6), 20:(2,8), 21:(3,7), 22:(3,5), 
             23:(4,4), 24:(4,6), 25:(4,8), 26:(5,7), 27:(5,5), 28:(6,6), 29:(6,8), 30:(7,5), 31:(8,5)}
    chan = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
    mask = np.zeros([9,9]).astype('bool')
    for i in range(len(chan)):
        if chan[i] in chan_name:
            x = order[i][0]
            y = order[i][1]
            mask[x][y] = True
    des = np.zeros([128,9,9]).astype('bool')
    for i in range(128):
        des[i] = mask
    return des
def mask_channel_lib(index, tag, conv_num, al_type, label_op, label):
    if label_op == 'total':
        order = np.load("/root/gradcam/channel_order/{}/conv{}_{}.npy".format(al_type,conv_num, tag)).astype('int32')
    elif label_op == 'label_0_1':
        order = np.load("/root/gradcam/channel_order/{}/conv{}_{}{}.npy".format(al_type,conv_num, tag, label)).astype('int32')
    else:
        print('error')
    des_chan = []
    #取一个样本的所有数据，即不对模型的输入改变。
    if index == 0:
        des_chan = chan
        mask = np.zeros([128, 9, 9]).astype('bool')
        mask = True
    #前4个通道值情况
    elif index == 1:
        for i in range(0, 4):
            des_chan.append(chan[order[31 - i]])
        #将对应通道以外的区域遮盖。
        mask = get_mask_(des_chan)
    #中4个通道值情况
    elif index == 2:
        for i in range(13, 17):
            des_chan.append(chan[order[31 - i]])
        mask = get_mask_(des_chan)
    #后4个通道值情况
    elif index == 3:
        for i in range(28, 32):
            des_chan.append(chan[order[31 - i]])
        mask = get_mask_(des_chan)
    #前8个通道值情况
    elif index == 4:
        for i in range(0, 8):
            des_chan.append(chan[order[31 - i]])
        mask = get_mask_(des_chan)
    #中8个通道值情况
    elif index == 5:
        for i in range(11, 19):
            des_chan.append(chan[order[31 - i]])
        mask = get_mask_(des_chan)
    #后8个通道值情况
    elif index == 6:
        for i in range(24, 32):
            des_chan.append(chan[order[31 - i]])
        mask = get_mask_(des_chan)
    #前10个通道值情况
    elif index == 7:
        for i in range(0, 10):
            des_chan.append(chan[order[31 - i]])
        mask = get_mask_(des_chan)
    #中10个通道值情况
    elif index == 8:
        for i in range(11, 21):
            des_chan.append(chan[order[31 - i]])
        mask = get_mask_(des_chan)
    #后8个通道值情况
    elif index == 9:
        for i in range(22, 32):
            des_chan.append(chan[order[31 - i]])
        mask = get_mask_(des_chan)
    #上1/3
    elif index == 10:
        des_chan = ['Fp1', 'Fp2', 'AF3', 'AF4','F7', 'F3', 'Fz', 'F4', 'F8']
        mask = np.zeros([128, 9, 9]).astype('bool')
        mask[:,0:3,:] = True
    #中1/3
    elif index == 11:
        des_chan = ['FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6']
        mask = np.zeros([128, 9, 9]).astype('bool')
        mask[:,3:6,:] = True
    #下1/3
    elif index == 12:
        des_chan = ['O1', 'O2', 'Oz', 'PO3','PO4', 'P7', 'P3', 'Pz', 'P4', 'P8']
        mask = np.zeros([128, 9, 9]).astype('bool')
        mask[:,6:9,:] = True
    #左半边
    elif index == 13:
        des_chan = ['Fp1', 'AF3', 'F7', 'F3', 'FC5', 'FC1','T7', 'C3','CP5', 'CP1','P7', 'P3', 'PO3', 'O1']
        mask = np.zeros([128, 9, 9]).astype('bool')
        mask[:,:,0:4] = True 
    #中间       
    elif index == 14:
        des_chan = ['Fz','Cz','Pz','Oz']
        mask = np.zeros([128, 9, 9]).astype('bool')
        mask[:,:,4] = True          
    elif index == 15:
        des_chan = ['Fp2','AF4','F4', 'F8', 'FC2', 'FC6','C4', 'T8', 'CP2', 'CP6', 'P4', 'P8', 'PO4','O2']
        mask = np.zeros([128, 9, 9]).astype('bool')
        mask[:,:,5:9] = True
    #右半边
    elif index == 16:
        mask = np.zeros([128, 9, 9]).astype('bool')
        mask[:, 0:3, 0:4] = True
    else:
        print("error")          
    return mask
def get_train_ans(conv_num, al_type, label_op, tag = "", index = 0):
    #加载模型和label,以及测试集的index
    if tag == "arousal":
        model_path = "/root/model/load_model/arousal/cross_2_test_acc_91.94010416666667.pkl"

        test_index = np.load("/root/model/train_test_info/arousal/test_index.npy")[2]
        label_path = "/root/lime/lime_info/val_arousal.npy"
    elif tag == "valence":
        model_path = "/root/model/load_model/valence/cross_5_test_acc_90.49479166666666.pkl"

        test_index = np.load("/root/model/train_test_info/valence/test_index.npy")[5]
        label_path = "/root/lime/lime_info/val_valence.npy"
    models = model_2d(num_class = 2, load_model = True, load_path = model_path)
    models.cuda()
    models.eval()
    label_val = np.load(label_path)
    False_val = 0
    false_val0 = 0
    false_val1 = 0
    total_val0 = 0
    total_val1 = 0
    total_val = 0
    des_acc_total = []
    des_acc_label0 = []
    des_acc_label1 = []
    for par in tqdm(range(32)):
        for ved in range(40):
            tem_label = -1
            #得到真实标签。
            if label_val[par][ved] <= 5:
                tem_label = 0
            else:
                tem_label = 1
            tem_false_label0 = 0
            tem_false_label1 = 0
            tem_total_sec = 0
            for sec in range(60):
                #判断此样本是否属于测试集。
                if (par * 40 * 60 + ved * 60 + sec + 1) in test_index:
                    #统计此视频中多少数据属于测试集
                    tem_total_sec = tem_total_sec + 1
                    #取到模型的输入数据
                    data = np.load("/root/data/DEAP/{}.npy".format(par * 40 * 60 + ved * 60 + sec + 1)).astype('float32')
                    if index == 0:
                        #此实验为查看测试集在模型上的准确率
                        output = models(torch.from_numpy(data).cuda().unsqueeze(0))
                    else:
                        #另外的实验。
                        output = models(torch.from_numpy(mask_channel(original_data = data, 
                                cha_name = mask_channel_lib(index  = index, tag = tag, conv_num = conv_num, al_type = al_type, label = tem_label, label_op = label_op,label = tem_label))).cuda().unsqueeze(0))
                    probs = output
                    preds = torch.max(probs,1)[1]
                    #模型预测与真实值不同时
                    if preds.cpu().numpy()[0] != tem_label:
                        False_val = False_val + 1
                        #真实标签为0时
                        if tem_label == 0:
                            #计数
                            false_val0 = false_val0 + 1
                            tem_false_label0 = tem_false_label0 + 1
                        #真实标签为1时
                        else:
                            false_val1 = false_val1 + 1
                            tem_false_label1 = tem_false_label1 + 1
                    if tem_label == 0:
                        total_val0 = total_val0 + 1
                    else:
                        total_val1 = total_val1 + 1
                    total_val = total_val + 1
            #统计此视频的准确率，并存储。
            if tem_label == 0 and tem_total_sec != 0:
                des_acc_label0.append((tem_total_sec - tem_false_label0) / tem_total_sec)
                des_acc_total.append((tem_total_sec - tem_false_label0) / tem_total_sec)
            if tem_label == 1 and tem_total_sec != 0:
                des_acc_label1.append((tem_total_sec - tem_false_label1) / tem_total_sec)
                des_acc_total.append((tem_total_sec - tem_false_label1) / tem_total_sec)
    #将得到的准确率转换为数组
    false_label0 = np.zeros(len(des_acc_label0))
    false_label1 = np.zeros(len(des_acc_label1))
    false_total = np.zeros((len(des_acc_total)))
    assert len(des_acc_label0) + len(des_acc_label1) == len(des_acc_total)
    for i in range(len(des_acc_label0)):
        false_label0[i] = des_acc_label0[i]
    for i in range(len(des_acc_label1)):
        false_label1[i] = des_acc_label1[i]
    for i in range(len(des_acc_total)):
        false_total[i] = des_acc_total[i]
    ans = np.zeros([6])
    #统计均值及标准差并存储。
    ans[0] = np.mean(false_total)
    ans[1] = np.mean(false_label0)
    ans[2] = np.mean(false_label1)
    ans[3] = np.std(false_total)
    ans[4] = np.std(false_label0)
    ans[5] = np.std(false_label1)
    print("[{}] Conv_{}  Exp_{} down".format(tag, conv_num, index))
    print("Total:{:.2%}[{}]  Label0:{:.2%}[{}]    Laebl1:{:.2%}[{}]".format(ans[0], total_val, ans[1], total_val0, ans[2], total_val1))
    print("Total:{}   Label0:{}    Laebl1:{}".format(ans[3], ans[4], ans[5]))
    np.save("/root/gradcam/acc/{}/{}_conv{}_exp{}.npy".format(al_type, tag, conv_num, index), ans)
if __name__ == "__main__":
    for al_type in ['Gradcam', 'Guided_Gradcam']:
        for order_type in ['total', 'label_0_1']:
            for tag in ['arousal','valence']:
                for conv_n in range(1, 2):
                    for i in range(16):
                    #遮罩通道并实验。
                        get_train_ans(tag = tag, index = i, al_type = al_type, label_op = order_type, conv_num = conv_n)


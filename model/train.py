from numpy.lib.function_base import append
from sklearn.utils import shuffle
import torch
from torch import nn, optim
from torch.nn.modules import module
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn, optim
import os
import numpy as np
import timeit
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

plt.rcParams['figure.figsize'] = (18.0, 10.0)

from data_class_2d import eeg_dataset2d
from models_2d import model_2d
from feature_extract import process_and_get_data

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

nepoch = 1000
num_class = 2
batch_size = 256
window_size = 128
save_pic_path = "/root/model/image/"
DEAP_path = "/root/DEAP/data_preprocessed_python/data_preprocessed_python/"
#生成的全体数据。
data_save_path = "/root/data/DEAP/"

def train_model(num_class = 2, lr = 1e-3, num_epoch = nepoch, extract_data = False, dest_label = " ", train_seed = 0, exp_index = 0):
    if extract_data:
        print("extract data")
        process_and_get_data(data_path = DEAP_path, load_path ="/root/model/")
        print("data load done")
    print("begin load data")
    time_1 = time.perf_counter()
    eeg_class = eeg_dataset2d(split = True,label_name = dest_label,path = "/root/model", save_path = data_save_path, train_seed = train_seed)
    total_train_data, total_test_data, total_train_label, total_test_label = eeg_class.return_data()
    time_2 = time.perf_counter()
    print("cost {}s load data".format(time_2 - time_1))

    total_runing_corrects = []
    total_test_corrects = []
    for i in range(10):
        print("start cross {}".format(i + 1))
        model = model_2d(num_class = num_class, load_model = False)

        train_params = [{'params':model_2d.get_1x_lr_paras(model), 'lr':lr},
                        {'params':model_2d.get_10x_lr_paras(model), 'lr':lr * 10}]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(train_params,lr = lr,momentum = 0.9, weight_decay = 1e-2)  
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma=0.1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #多gpu训练
        device_count = torch.cuda.device_count()
        if device_count > 1:
            divece_ids = [0, 1]
            model = nn.DataParallel(model, device_ids = divece_ids)
        model.to(device)
        criterion.to(device)
        train_dataloader = DataLoader(eeg_dataset2d(train_data = total_train_data[i],train_label = total_train_label[i], label_name = dest_label,data_type = "train"), shuffle = True, batch_size = batch_size, num_workers = 4)
        test_dataloader = DataLoader(eeg_dataset2d(test_data = total_test_data[i],test_label = total_test_label[i], label_name = dest_label, data_type = "test"), batch_size = batch_size, num_workers = 4)
        traintest_loaders = {'train':train_dataloader, 'test':test_dataloader}
        traintest_sizes = {x: len(traintest_loaders[x].dataset) for x in ['train', 'test']}

        plt_acc = []
        plt_eporch = []
        plt_test_acc = []
        plt_train_loss = []
        plt_lr = []
        plt_test_loss = []
        pre_acc = 0
        is_stop = False
        for epoch in range(num_epoch):
            if is_stop == True:
                break
            for phase in ['train', 'test']:
                epoch_loss = 0.0
                epoch_acc = 0.0
                running_loss = 0.0
                running_corrects = 0.0

                if phase == 'test':
                    model.eval()
                for inputs, labels in tqdm(traintest_loaders[phase]):
                    inputs = Variable(inputs, requires_grad = True).to(device)
                    label_ = labels.squeeze(1)
                    label = Variable(label_).to(device)
                    if phase == 'train':
                        optimizer.zero_grad()
                        outputs = model(inputs)
                    else:
                        with torch.no_grad():
                            outputs = model(inputs)
                    
                    probs = outputs
                    preds = torch.max(probs, 1)[1]
                    label = label.long()              #转换为long型数据
                    loss = criterion(outputs, label)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)  
                    running_corrects += torch.sum(preds == label.data)

                if phase == 'train':        
                    scheduler.step()
                    model.train()
                
                epoch_loss = running_loss / traintest_sizes[phase]
                epoch_acc = running_corrects.double() / traintest_sizes[phase]

                if phase == 'train':
                    plt_acc.append(epoch_acc.cpu())
                    plt_eporch.append(epoch)
                    plt_train_loss.append(epoch_loss)
                if phase == 'test':
                    plt_test_acc.append(epoch_acc.cpu())
                    plt_test_loss.append(epoch_loss)
                print("dataset {} [{}] Epoch: {}/{} Loss: {} Acc: {}".format(i,phase, epoch+1, nepoch, epoch_loss, epoch_acc))
                stop_time = timeit.default_timer()
            if (epoch >= 60 and epoch_acc.cpu().item() > 0.90) and (pre_acc > 0.90) and ((epoch_acc.cpu().item() - pre_acc < 0.0008) or (epoch_acc.cpu().item() - pre_acc > -0.0008))or(epoch >= 100 and epoch_acc.cpu().item() >= 0.89 and pre_acc >= 0.89):
                torch.save(model.state_dict(), "/root/model/model_save/{}/cross_{}_test_acc_{}.pkl".format(dest_label,i, epoch_acc.cpu().item() * 100))
                is_stop = True
            if epoch == 9 or(plt_acc[epoch] == 1 and plt_test_acc[epoch] == 1):
                total_runing_corrects.append(plt_acc[epoch])
                total_test_corrects.append(plt_test_acc[epoch])
            pre_acc = epoch_acc.cpu().item()
            plt_lr.append(optimizer.param_groups[0]["lr"])
            ax1 = plt.subplot(3, 2, 1)
            plt.plot(plt_eporch, plt_train_loss, color = 'red')
            plt.xlabel('epoch')
            plt.ylabel('train loss')
            ax2 = plt.subplot(3, 2, 2)
            plt.plot(plt_eporch, plt_test_loss, color = 'blue')
            plt.xlabel('epoch')
            plt.ylabel('lr')
            ax3 = plt.subplot(3, 2, 3)
            plt.plot(plt_eporch, plt_acc, color = 'red')
            plt.xlabel('epoch')
            plt.ylabel('train acc')
            ax4 = plt.subplot(3, 2, 4)
            plt.plot(plt_eporch, plt_test_acc, color = 'blue')
            plt.xlabel('epoch')
            plt.ylabel('test acc')
            ax5 = plt.subplot(3, 2, 5)
            plt.plot(plt_eporch, plt_lr, color = 'green')
            plt.xlabel('epoch')
            plt.ylabel('lr')
            plt.savefig(save_pic_path+"/{}/cross{}_out_drop_sgd _5e-1_0.5_lr{}paralr_batch{}.png".format(dest_label,i + 1,lr, batch_size))
            plt.close()
    
if __name__ == '__main__':
    for label_name in ['arousal','valence']:
        train_model(dest_label = label_name, extract_data = False, train_seed = 2020)

    



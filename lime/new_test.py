from xlwt import *
import numpy as np
import os
import _pickle as cPickle

if not os.path.exists("/root/lime/lime_info/val_arousal.npy"):
    chan = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
    arousal_data = np.zeros([32,40])
    valence_data = np.zeros([32,40])
    for i in tqdm(range(32)):
        for ve in range(40): #40
            if i < 9:
                dest_file = "/root/DEAP/data_preprocessed_python/data_preprocessed_python/" + "s0" + str(i + 1) + ".dat"
            else:
                dest_file = "/root/DEAP/data_preprocessed_python/data_preprocessed_python/" + "s" + str(i + 1) + ".dat"
    
            data = cPickle.load(open(dest_file, 'rb'), encoding = 'bytes')
            arousal_data[i][ve] = data[b'labels'][ve][1]
            valence_data[i][ve] = data[b'labels'][ve][0]
    np.save("/root/lime/lime_info/val__arousal.npy",arousal_data)
    np.save("/root/lime/lime_info/val__valence.npy", valence_data)

for label in ['arousal', 'valence']:
    if label == 'arousal':
        test_index = np.load("/root/model/train_test_info/{}/test_index.npy".format(label))[2]
    elif label == 'valence':
        test_index = np.load("/root/model/train_test_info/{}/test_index.npy".format(label))[5]
    total_label = np.load("/root/lime/lime_info/val_{}.npy".format(label))
    
    for exp in range(1,3):
        total_test_data_weight_0 = np.zeros([40,32,60,3])
        total_test_data_0_1_value_0 = np.zeros([40,32,60,3])
        total_test_data_weight_1 = np.zeros([40,32,60,3])
        total_test_data_0_1_value_1 = np.zeros([40,32,60,3])
        for ved in range(40):
            data = np.load("/root/lime/lime_data/epx{}/{}/vedio_{}.npy".format(exp,label,ved))
            for par in range(32):
                if total_label[par][ved] <= 5:
                    true_label = 0
                else:
                    true_label = 1
                for sec in range(60):
                    if (par * 40 * 60 + ved * 60 + sec + 1) in test_index:
                        if exp == 1:
                            if true_label == 0:
                                total_test_data_weight_0[ved][par][sec][0] += data[par][sec][0][0]
                                total_test_data_weight_0[ved][par][sec][1] += data[par][sec][3][0]
                                total_test_data_weight_0[ved][par][sec][2] += data[par][sec][6][0]
                                total_test_data_0_1_value_0[ved][par][sec][0] += 1 if (data[par][sec][0][0]!= 0) else 0
                                total_test_data_0_1_value_0[ved][par][sec][1] += 1 if (data[par][sec][3][0]!= 0) else 0
                                total_test_data_0_1_value_0[ved][par][sec][2] += 1 if (data[par][sec][6][0]!= 0) else 0
                            else:
                                total_test_data_weight_1[ved][par][sec][0] += data[par][sec][0][0]
                                total_test_data_weight_1[ved][par][sec][1] += data[par][sec][3][0]
                                total_test_data_weight_1[ved][par][sec][2] += data[par][sec][6][0]
                                total_test_data_0_1_value_1[ved][par][sec][0] += 1 if (data[par][sec][0][0]!= 0) else 0
                                total_test_data_0_1_value_1[ved][par][sec][1] += 1 if (data[par][sec][3][0]!= 0) else 0
                                total_test_data_0_1_value_1[ved][par][sec][2] += 1 if (data[par][sec][6][0]!= 0) else 0
                        elif exp == 2:
                            if true_label == 0:
                                total_test_data_weight_0[ved][par][sec][0] += data[par][sec][0][0]
                                total_test_data_weight_0[ved][par][sec][1] += data[par][sec][0][4]
                                total_test_data_weight_0[ved][par][sec][2] += data[par][sec][0][5]
                                total_test_data_0_1_value_0[ved][par][sec][0] += 1 if (data[par][sec][0][0]!= 0) else 0
                                total_test_data_0_1_value_0[ved][par][sec][1] += 1 if (data[par][sec][0][4]!= 0) else 0
                                total_test_data_0_1_value_0[ved][par][sec][2] += 1 if (data[par][sec][0][5]!= 0) else 0
                            else:
                                total_test_data_weight_1[ved][par][sec][0] += data[par][sec][0][0]
                                total_test_data_weight_1[ved][par][sec][1] += data[par][sec][0][4]
                                total_test_data_weight_1[ved][par][sec][2] += data[par][sec][0][5]
                                total_test_data_0_1_value_1[ved][par][sec][0] += 1 if (data[par][sec][0][0]!= 0) else 0
                                total_test_data_0_1_value_1[ved][par][sec][1] += 1 if (data[par][sec][0][4]!= 0) else 0
                                total_test_data_0_1_value_1[ved][par][sec][2] += 1 if (data[par][sec][0][5]!= 0) else 0
                        else:
                            print("error")
        test_ved_data_weight_0 = np.zeros([40,3])
        test_ved_data_0_1_value_0 = np.zeros([40,3])
        test_ved_data_weight_mean_0 = np.zeros([40,3])
        test_ved_data_weight_std_0 = np.zeros([40,3])
        test_ved_data_weight_1 = np.zeros([40,3])
        test_ved_data_0_1_value_1 = np.zeros([40,3])
        test_ved_data_weight_mean_1 = np.zeros([40,3])
        test_ved_data_weight_std_1 = np.zeros([40,3])

        for ved in range(40):
            for par in range(32):
                for sec in range(60):
                    test_ved_data_weight_0[ved] += total_test_data_weight_0[ved][par][sec]
                    test_ved_data_0_1_value_0[ved] += total_test_data_0_1_value_0[ved][par][sec]
                    test_ved_data_weight_1[ved] += total_test_data_weight_1[ved][par][sec]
                    test_ved_data_0_1_value_1[ved] += total_test_data_0_1_value_1[ved][par][sec]
        des_0 = total_test_data_weight_0.transpose(0,3,1,2)
        des_1 = total_test_data_weight_1.transpose(0,3,1,2)
        for ved in range(40):
            for col in range(3):
                if len(np.where(des_0[ved][col] != 0)[0]) != 0:
                    test_ved_data_weight_mean_0[ved][col] = np.mean(des_0[ved][col][np.where(des_0[ved][col] != 0)])
                    test_ved_data_weight_std_0[ved][col] = np.std(des_0[ved][col][np.where(des_0[ved][col] != 0)])
                if len(np.where(des_1[ved][col] != 0)[0]) != 0:
                    test_ved_data_weight_mean_1[ved][col] = np.mean(des_1[ved][col][np.where(des_1[ved][col] != 0)])
                    test_ved_data_weight_std_1[ved][col] = np.std(des_1[ved][col][np.where(des_1[ved][col] != 0)])
        for des in ['test']:
            file = Workbook(encoding = 'utf-8')
            table = file.add_sheet('data')
            if exp == 1:
                table.write(0,0,"weight")
                table.write(0,1,"上")
                table.write(0,2,"中")
                table.write(0,3,"下")
            elif exp == 2:
                table.write(0,0,"weight")
                table.write(0,1,"左")
                table.write(0,2,"中")
                table.write(0,3,"右")
            for ved in range(8,9):
                num_index = 1
                for par in range(32):
                    for sec in range(60):
                        if (total_test_data_weight_0[ved][par][sec][0] != 0 or total_test_data_weight_0[ved][par][sec][1] != 0 or 
                            total_test_data_weight_0[ved][par][sec][2] != 0):
                            table.write(num_index, 0, "ved_{}_info_{}".format(ved,num_index))
                            for col in range(3):
                                table.write(num_index, col + 1,  total_test_data_weight_0[ved][par][sec][col])
                            num_index += 1
            if exp == 1: 
                file.save('/root/lime/lime_info/{}0_exp{}.xls'.format(label,exp))
            elif exp == 2:
                file.save('/root/lime/lime_info/{}0_exp{}.xls'.format(label,exp))
        for des in ['test']:
            file = Workbook(encoding = 'utf-8')
            table = file.add_sheet('data')
            if exp == 1:
                table.write(0,0,"weight")
                table.write(0,1,"上")
                table.write(0,2,"中")
                table.write(0,3,"下")
            elif exp == 2:
                table.write(0,0,"weight")
                table.write(0,1,"左")
                table.write(0,2,"中")
                table.write(0,3,"右")
            for ved in range(22,23):
                num_index = 1
                for par in range(32):
                    for sec in range(60):
                        if (total_test_data_weight_1[ved][par][sec][0] != 0 or total_test_data_weight_1[ved][par][sec][1] != 0 or 
                            total_test_data_weight_1[ved][par][sec][2] != 0):
                            table.write(num_index, 0, "ved_{}_info_{}".format(ved,num_index))
                            for col in range(3):
                                table.write(num_index, col + 1,  total_test_data_weight_1[ved][par][sec][col])
                            num_index += 1
            if exp == 1: 
                file.save('/root/lime/lime_info/{}1_exp{}.xls'.format(label,exp))
            elif exp == 2:
                file.save('/root/lime/lime_info/{}1_exp{}.xls'.format(label,exp))


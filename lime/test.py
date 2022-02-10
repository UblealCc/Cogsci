from xlwt import *
import numpy as np
import os
import _pickle as cPickle

for label in ['arousal', 'valence']:
    if label == 'arousal':
        test_index = np.load("/root/model/train_test_info/{}/test_index.npy".format(label))[2]
    elif label == 'valence':
        test_index = np.load("/root/model/train_test_info/{}/test_index.npy".format(label))[5]
    for exp in range(1,3):
        total_test_data_weight = np.zeros([40,32,60,3])
        total_test_data_0_1_value = np.zeros([40,32,60,3])
        for ved in range(40):
            data = np.load("/root/lime/lime_data/epx{}/{}/vedio_{}.npy"
                                .format(exp,label,ved))
            for par in range(32):
                for sec in range(60):
                    if (par * 40 * 60 + ved * 60 + sec + 1) in test_index:
                        if exp == 1:
                            total_test_data_weight[ved][par][sec][0] += data[par][sec][0][0]
                            total_test_data_weight[ved][par][sec][1] += data[par][sec][3][0]
                            total_test_data_weight[ved][par][sec][2] += data[par][sec][6][0]
                            total_test_data_0_1_value[ved][par][sec][0] += 1 if (data[par][sec][0][0]!= 0) else 0
                            total_test_data_0_1_value[ved][par][sec][1] += 1 if (data[par][sec][3][0]!= 0) else 0
                            total_test_data_0_1_value[ved][par][sec][2] += 1 if (data[par][sec][6][0]!= 0) else 0
                        elif exp == 2:
                            total_test_data_weight[ved][par][sec][0] += data[par][sec][0][0]
                            total_test_data_weight[ved][par][sec][1] += data[par][sec][0][4]
                            total_test_data_weight[ved][par][sec][2] += data[par][sec][0][5]
                            total_test_data_0_1_value[ved][par][sec][0] += 1 if (data[par][sec][0][0]!= 0) else 0
                            total_test_data_0_1_value[ved][par][sec][1] += 1 if (data[par][sec][0][4]!= 0) else 0
                            total_test_data_0_1_value[ved][par][sec][2] += 1 if (data[par][sec][0][5]!= 0) else 0
                        else:
                            print("error")
        test_ved_data_weight = np.zeros([40,3])
        test_ved_data_0_1_value = np.zeros([40,3])
        test_ved_data_weight_mean = np.zeros([40,3])
        test_ved_data_weight_std = np.zeros([40,3])
        for ved in range(40):
            for par in range(32):
                for sec in range(60):
                    test_ved_data_weight[ved] += total_test_data_weight[ved][par][sec]
                    test_ved_data_0_1_value[ved] += total_test_data_0_1_value[ved][par][sec]
        des = total_test_data_weight.transpose(0,3,1,2)
        for ved in range(40):
            for col in range(3):
                if len(np.where(des[ved][col] != 0)[0]) != 0:
                    test_ved_data_weight_mean[ved][col] = np.mean(des[ved][col][np.where(des[ved][col] != 0)])
                    test_ved_data_weight_std[ved][col] = np.std(des[ved][col][np.where(des[ved][col] != 0)])

        for des in ['test']:
            file = Workbook(encoding = 'utf-8')
            #指定file以utf-8的格式打开
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
            for ved in range(40):
                table.write(ved + 1, 0, "ved_{}".format(ved))
                for col in range(3):
                    table.write(ved + 1, col + 1,  test_ved_data_weight[ved][col])
            if exp == 1:
                table.write(0,4,"0_1_value")
                table.write(0,5,"上")
                table.write(0,6,"中")
                table.write(0,7,"下")
            elif exp == 2:
                table.write(0,4,"0_1_value")
                table.write(0,5,"左")
                table.write(0,6,"中")
                table.write(0,7,"右")  
            for ved in range(40):           
                table.write(ved + 1, 4, "ved_{}".format(ved))
                for col in range(3):
                    table.write(ved  + 1, col + 1 + 4,test_ved_data_0_1_value[ved][col])
            if exp == 1:
                table.write(0,8,"weight_mean")
                table.write(0,9,"上_mean")
                table.write(0,10,"中_mean")
                table.write(0,11,"下_mean")
            elif exp == 2:
                table.write(0,8,"weight_mean")
                table.write(0,9,"左_mean")
                table.write(0,10,"中_mean")
                table.write(0,11,"右_mean")  
            for ved in range(40):
                table.write(ved + 1, 8, "ved_{}".format(ved))
                for col in range(3):
                    table.write(ved + 1, col + 1 + 8, test_ved_data_weight_mean[ved][col])
            if exp == 1:
                table.write(0,12,"weight_std")
                table.write(0,13,"上_std")
                table.write(0,14,"中_std")
                table.write(0,15,"下_std")
            elif exp == 2:
                table.write(0,12,"weight_std")
                table.write(0,13,"左_std")
                table.write(0,14,"中_std")
                table.write(0,15,"右_std")  
            for ved in range(40):
                table.write(ved + 1, 12, "ved_{}".format(ved))
                for col in range(3):
                    table.write(ved + 1, col + 1 + 12, test_ved_data_weight_std[ved][col])
            if exp == 1: 
                file.save('/root/lime/lime_info/{}_total_exp{}.xls'.format(label,exp))
            elif exp == 2:
                file.save('/root/lime/lime_info/{}_total_exp{}.xls'.format(label,exp))


from PIL import Image
import torch
from lime import lime_image
import numpy as np
from skimage.segmentation import mark_boundaries
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model_2d import model_2d_with_pool as model_2d
import os
import seaborn as sns
from scipy.interpolate import griddata

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_path = "/root/model/load_model/arousal/cross_2_test_acc_91.94010416666667.pkl"
# model_path = "/root/model/load_model/valence/cross_5_test_acc_90.49479166666666.pkl"

models = model_2d(num_class = 2, load_model = True, load_path = model_path)

def normal_data(data):
    max = data.max()
    min = data.min()
    data = (data - min) / (max - min)
    return data

def load_data(path, index):
    load_path = path + str(index) + ".npy"
    return np.load(load_path).astype('float32')

def batch_predict(data, models = models):
    if data.shape ==(10, 9, 9, 128):
        datas = data.transpose(0,3,1,2).astype('float32')
        batch = torch.tensor(datas).contiguous()
    elif data.shape ==(10, 128, 32, 3):
        datas = np.zeros([10, 128, 9, 9])
        for i in range(10):
            datas[i] = transform_to_matric((data.transpose(0,3,1,2))[i][0])
        batch = torch.tensor(datas.astype('float32'))
    elif data.shape == (128,9,9):
        datas = data
        batch = torch.tensor(datas).unsqueeze(0)
    elif data.shape == (11, 128, 32, 3):
        datas = np.zeros([11, 128, 9, 9])
        for i in range(11):
            datas[i] = transform_to_matric((data.transpose(0,3,1,2))[i][0])
        batch = torch.tensor(datas.astype('float32'))
    else:
        print("the input shape is :")
        print(data.shape)
    models.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models.to(device)
    batch = batch.to(device)
    logits = models(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def gen_2d_planes(features, n_gridpoints):
    locs = np.loadtxt(open("/root/model/loc.txt","rb"),delimiter=",",skiprows=0)
    nElectrodes = locs.shape[0]  # Number of electrodes
    assert features.shape[1] % nElectrodes == 0
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints * 1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints * 1j,
                     ]
    temp_interp = []
    temp_interp.append(np.zeros([n_gridpoints, n_gridpoints]))
    temp_interp[0][ :, :] = griddata(locs, features[0, :], (grid_x, grid_y),
                                               method='cubic', fill_value=0)

    return np.array(temp_interp).squeeze()

def lime_process(data, exp_index):
    explainer = lime_image.LimeImageExplainer()

    input_data = data.transpose(1,2,0)
    explanation = explainer.explain_instance(input_data, 
                                         batch_predict,
                                         exp_index = exp_index,
                                         top_labels=1, 
                                         hide_color=0, 
                                         batch_size = 10,
                                         num_samples=100) 

    return explanation

def get_mask(explanation):
    temp1, mask1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)
    return mask1
def fix_pic_b(data):
    des = np.zeros([128,9,9])
    index = -1
    for row in range(12):
        for col in range(12):
            index = index + 1
            if index == 128:
                break
            else:
                des[index][0:9, 0:9] = data[row*9:row*9+9, col*9:col*9 + 9]
        if index == 128:
            break
    return des

def fix_pic_a(data):
    des = np.zeros([12*9,12*9])
    index = -1
    for row in range(12):
        for col in range(12):
            index = index + 1
            if index == 128:
                break
            else:
                des[row*9:(row*9+9),col*9:(col*9 + 9)] = data[index][0:9,0:9]
        if index == 128:
            break
    return des

def trans_to_brain_image(data, save_name, save_path, index):   
    new_data = transform_to_matric(data)
    for i in range(128):
        x = []
        y = []
        for row in range(9):
            for col in range(9):
                if new_data[i][row][col] == 1:
                    x.append(row)
                    y.append(col)
        plt.scatter(x,y,s = 75, alpha = 0.5,color = 'red')
        if not os.path.exists(save_path +"/" + save_name):
            os.mkdir(save_path + "/" + save_name)
        plt.savefig(save_path +"/"+ save_name +"/"+ str(index) + "_s_"+str(i) + "sam.png")


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

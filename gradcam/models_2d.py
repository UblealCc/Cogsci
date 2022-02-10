import torch
import torch.nn as nn
from torch.serialization import load

class model_2d(nn.Module):

    def __init__(self, num_class, load_model = False, load_path = ""):
        self.load_path = load_path
        super(model_2d, self).__init__()
        self.conv1 = nn.Conv2d(128 ,128, kernel_size=(3, 3), stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride = 2, padding = 1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.7)
        self.fc1 = nn.Linear(512*9, 1024)
        self.fc3 = nn.Linear(1024,1024)
        self.fc2 = nn.Linear(1024, num_class)
        self.soft = nn.Softmax(dim = 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        if load_model == True:
            self._load_module()

    def _load_module(self):
        corresp_name = {
                        "module.conv1.weight": "conv1.weight",
                        "module.conv1.bias": "conv1.bias",

                        "module.conv2.weight": "conv2.weight",
                        "module.conv2.bias": "conv2.bias",
                        
                        "module.conv3.weight": "conv3.weight",
                        "module.conv3.bias": "conv3.bias",

                        "module.conv4.weight": "conv4.weight",
                        "module.conv4.bias": "conv4.bias",

                        "module.conv5.weight": "conv5.weight",
                        "module.conv5.bias": "conv5.bias",

                        "module.pool1.weight": "pool1.weight",
                        "module.pool1.bias": "pool1.bias",

                        "module.pool2.weight": "pool2.weight",
                        "module.pool2.bias": "pool2.bias",

                        "module.fc1.weight": "fc1.weight",
                        "module.fc1.bias": "fc1.bias",

                        "module.fc2.weight": "fc2.weight",
                        "module.fc2.bias": "fc2.bias",

                        "module.fc3.weight": "fc3.weight",
                        "module.fc3.bias": "fc3.bias",
                        }
        
        p_dict = torch.load(self.load_path)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))

        x = x.view(-1, 512*9)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        logist = self.fc2(x)
        logist = self.soft(logist)


        return logist
    def get_1x_lr_paras(model):
        b = [model.conv1, model.conv2, model.conv3, model.conv4,model.fc1, model.fc3]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_paras(model):
        b = [model.fc2]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k  

class model_2d_remove(nn.Module):

    def __init__(self, num_class, load_model = False, load_path = ""):
        self.load_path = load_path
        super(model_2d, self).__init__()
        self.conv1 = nn.Conv2d(128 ,128, kernel_size=(3, 3), stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride = 2, padding = 1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.7)
        self.fc1 = nn.Linear(512*9, 1024)
        self.fc3 = nn.Linear(1024,1024)
        self.fc2 = nn.Linear(1024, num_class)
        self.soft = nn.Softmax(dim = 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        if load_model == True:
            self._load_module()

    def _load_module(self):
        corresp_name = {
                        "module.conv1.weight": "conv1.weight",
                        "module.conv1.bias": "conv1.bias",

                        "module.conv2.weight": "conv2.weight",
                        "module.conv2.bias": "conv2.bias",
                        
                        "module.conv3.weight": "conv3.weight",
                        "module.conv3.bias": "conv3.bias",

                        "module.conv4.weight": "conv4.weight",
                        "module.conv4.bias": "conv4.bias",

                        "module.conv5.weight": "conv5.weight",
                        "module.conv5.bias": "conv5.bias",

                        "module.pool1.weight": "pool1.weight",
                        "module.pool1.bias": "pool1.bias",

                        "module.pool2.weight": "pool2.weight",
                        "module.pool2.bias": "pool2.bias",

                        "module.fc1.weight": "fc1.weight",
                        "module.fc1.bias": "fc1.bias",

                        "module.fc2.weight": "fc2.weight",
                        "module.fc2.bias": "fc2.bias",

                        "module.fc3.weight": "fc3.weight",
                        "module.fc3.bias": "fc3.bias",
                        }
        
        p_dict = torch.load(self.load_path)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))

        x = x.view(-1, 512*9)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        logist = self.fc2(x)

        return logist
    def get_1x_lr_paras(model):
        b = [model.conv1, model.conv2, model.conv3, model.conv4,model.fc1, model.fc3]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_paras(model):
        b = [model.fc2]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k  
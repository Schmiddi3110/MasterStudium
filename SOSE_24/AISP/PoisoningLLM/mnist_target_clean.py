# %% [markdown]
# # Data Poisoning
# 
# This code is originally based on an exmaple notebook from ART:
# https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_witches_brew_pytorch.ipynb
# 
# ## References 
# [1] Witches' Brew arXiv:2009.02276v2
# [2] https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_witches_brew_pytorch.ipynbx



# %%
import numpy as np
import os, sys
import tqdm
from tqdm import trange

# %%

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)


# %%

from art.utils import load_mnist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # mps does not work


# Model 
class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=n_classes)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


def _create_model(model_name, n_classes):
    model = Net(n_classes)
    
    return model



def data_processing_mnist(classes=None, device=device):
    (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)

    x_train = np.transpose(x_train, [0, 3,1,2])
    x_test = np.transpose(x_test, [0, 3,1,2])

    min_ = (min_-mean)/(std+1e-7)
    max_ = (max_-mean)/(std+1e-7)


    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    if classes is not None:
        idx_tr = np.argwhere([y in classes for y in y_train]).ravel()
        idx_te = np.argwhere([y in classes for y in y_test]).ravel()
        
        x_train = x_train[idx_tr]
        y_train = y_train[idx_tr]

        x_test = x_test[idx_te]
        y_test = y_test[idx_te]

        new_y_train = []
        new_y_test = []
        for i,y in enumerate(y_train):
                new_y_train.append(classes.index(y))
        for i,y in enumerate(y_test):
                new_y_test.append(classes.index(y))

        y_train = np.array(new_y_train)
        y_test = np.array(new_y_test)

   
    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device) # transform to torch tensor
    x_tensor_test = torch.tensor(x_test, dtype=torch.float32, device=device) # transform to torch tensor
    
    y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)

    y_tensor_test = torch.tensor(y_test, dtype=torch.long, device=device)

    return x_tensor, y_tensor, x_tensor_test, y_tensor_test


# TO GET THE MODEL RUNNING:

MODEL_NAME = "mnist_4_dc30"

n_classes = 4
model = _create_model(MODEL_NAME, n_classes)

model_checkpoint_path = './extract_target_%s_mnist.pt' % MODEL_NAME
if os.path.isfile(model_checkpoint_path):
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('loaded model checkpoint')
else:
    raise FileNotFoundError("Checkpoint file not found, check your path.")

x_train, y_train, x_test, y_test = data_processing_mnist()

# Test
prediction = model(x_test)



# %%

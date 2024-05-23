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
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # mps does not work


# Model 

# import the model, training routines you need from your victim's code...


print("Model and data preparation done.")

model_art = PyTorchClassifier() # TODO 



# SELECT TARGET IMAGES 
# ATTACK GOAL
# - select a source class ID (0...9) and a target class id 
# - select a bunch of images belonging to the source class and try to get the victim's model to classify them as target class.target_images
# - try several source/target combinations  
# - use Gradient Matching Attack

SOURCE = 
TARGET = 



x_trigger = 
y_trigger = 


epsilson = 
percent_poison = 0.01
attack = 

# EVALUATE YOUR ATTACK
# - assume you can put 1% (percent_poison=0.01) of poisoned images into the victim's training data. The remainder of the training data is unchanged.
# - train a model to see how successful your attack will be then
# - how many of your selected target images get correctly misclassified (according to your attack)?
# - can you optimize the attack to increase your success?
# - would you be more successful with 2% poison images? How about 0.1%?



# %%

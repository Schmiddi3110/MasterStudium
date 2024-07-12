# # Data Poisoning
# 
# This code is originally based on an exmaple notebook from ART:
# https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_witches_brew_pytorch.ipynb
# 
# ## References 
# [1] Witches' Brew arXiv:2009.02276v2
# [2] https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_witches_brew_pytorch.ipynbx




import numpy as np
import os, sys
import tqdm
from tqdm import trange
from art.utils import load_cifar10
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import timm

def _create_model(model_name, classes):
    model = timm.create_model(model_name, pretrained=True) # TODO Consider pretrained=True and False
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, classes) # adapt final layer to match the 10 classes of CIFAR-10

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # mps does not work

def data_processing(x_train, y_train, x_test, y_test, upsampler, device=device):
    
    if upsampler is None:
        x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device) # transform to torch tensor
        x_tensor_test = torch.tensor(x_test, dtype=torch.float32, device=device) # transform to torch tensor

    else:
        _x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device) # transform to torch tensor
        x_tensor = upsampler(_x_tensor)
        del _x_tensor

        _x_tensor_test = torch.tensor(x_test, dtype=torch.float32, device=device) # transform to torch tensor
        x_tensor_test = upsampler(_x_tensor_test)
        del _x_tensor_test

    y_train = np.argmax(y_train, axis=1)
    y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)

    y_test = np.argmax(y_test, axis=1)
    y_tensor_test = torch.tensor(y_test, dtype=torch.long, device=device)

    return x_tensor, y_tensor, x_tensor_test, y_tensor_test

def _testAccuracy(model, test_loader, transform=None, max_steps=10):
    model_was_training = model.training
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            if transform is not None:
                images = transform(images)

            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    if model_was_training:
      model.train()
    return(accuracy)


def train_model(model, optimizer, loss_fn, x_train, y_train, x_test, y_test, x_trigger=None, y_trigger=None, transform=None, batch_size=128, epochs=80):
    
    model.to(device)

    dataset_train = TensorDataset(x_train.to(device), y_train.to(device)) # create your datset
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)

    dataset_test = TensorDataset(x_test.to(device), y_test.to(device)) # create your datset
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

    iter = trange(epochs)
    for _ in iter:
        running_loss = 0.0
        total = 0
        accuracy = 0
        for _, data in enumerate(dataloader_train, 0):
            inputs, labels = data
            optimizer.zero_grad()

            if transform is not None:
                outputs = model(transform(inputs))
            else:
                outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            running_loss += loss.item()
        train_accuracy = (accuracy / total)
        print ("train acc", train_accuracy)

        if x_trigger is not None:
            y_ = model(x_trigger)
            y_ = F.softmax(y_, dim=-1)[0]
            output_target = y_.detach().cpu().numpy()[y_trigger]
            iter.set_postfix({'acc': train_accuracy, 'target': output_target})
            tqdm.tqdm.write(str(output_target))
        else:
            iter.set_postfix({'acc': train_accuracy})
    test_accuracy = _testAccuracy(model, dataloader_test, transform)
    print("Final test accuracy: %f" % test_accuracy)

    del dataset_train, dataloader_train
    del dataset_test, dataloader_test

    return model






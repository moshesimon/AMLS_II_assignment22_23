import timm
import torch.nn as nn
import torch
from config import *


class Net(nn.Module):
    def __init__(self, base_model):
        super(Net, self).__init__()
        self.base_model = base_model
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = self.base_model(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
def get_model():

    # define the model
    print("Creating model...")
    base_model = timm.create_model('efficientnetv2_s', num_classes = 1280, pretrained = True, in_chans = 1)
    for param in base_model.parameters():
        param.requires_grad = False

    model = Net(base_model)
    return model
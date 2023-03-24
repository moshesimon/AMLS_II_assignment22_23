import timm
import torch.nn as nn
import torch
from config import *


class Net(nn.Module):
    def __init__(self, model_name, base_model, num_meta_features):
        super(Net, self).__init__()
        self.base_model = base_model
        if model_name == 'efficientnet':
            self.fc1 = nn.Linear(1280, 512)
        else:
            self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(72, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.meta_nn = nn.Sequential(
            nn.Linear(num_meta_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )


    def forward(self, x, metadata):
        x = self.base_model(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        metadata = self.meta_nn(metadata)
        
        combined = torch.cat((x, metadata), dim=1)
        
        x = self.fc3(combined)

        return x
    
def get_model(model_name,num_meta_features):

    # define the model
    print("Creating model...")
    if model_name == 'efficientnet':
        base_model = timm.create_model('efficientnetv2_s', num_classes=1280, pretrained=True, in_chans=1)
    elif model_name == 'resnet50':
        base_model = timm.create_model('resnet50', num_classes=2048, pretrained=True, in_chans=1)
    else:
        raise ValueError("Invalid model name. Choose 'efficientnet' or 'resnet50'.")

    for param in base_model.parameters():
        param.requires_grad = False

    model = Net(model_name, base_model, num_meta_features)
    return model
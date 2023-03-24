from torch.utils.data import Dataset, random_split
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from config import *


class RSNA_Dataset(Dataset):
    def __init__(self, path, labels, transform=None):
        self.path = path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        img_name = os.path.join(self.path,f"{self.labels.loc[index, 'patient_id']}_{self.labels.loc[index, 'image_id']}.png")
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(self.labels.loc[index, 'cancer'])
        metadata = {
            'age': (self.labels.loc[index, 'age'] - 26) / 63, 
            'implant': self.labels.loc[index, 'implant'],
        }
        metadata = torch.tensor([metadata[key] for key in metadata], dtype=torch.float32)
        return img, metadata, label
    
def get_data_loaders():

    print("Getting data...")
    labels = pd.read_csv(labels_augmented_dir)
    dataset = RSNA_Dataset(images_augmented_dir, labels, transform)
    print("Data loaded!")

    # split the dataset into train and validation
    print("Splitting data...")
    #[85453, 21363]
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train, val = random_split(dataset,[train_len, val_len],torch.Generator().manual_seed(42))# 4:1 split

    # create the dataloaders
    print("Creating dataloaders...")
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=64, num_workers=1, pin_memory=True)

    return train_loader, val_loader
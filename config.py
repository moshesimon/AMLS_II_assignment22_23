import os
import random

from torchvision import transforms
import torch

IMAGE_SIZE = 1024

base_dir = os.path.dirname(__file__)
dataset_dir = "/scratch/zceemsi/AMLS_II_assignment22_23/Datasets"

local_dataset_dir = os.path.join(base_dir, "Datasets")
rsna_256_dir = os.path.join(local_dataset_dir, "rsna-breast-cancer-256-pngs")

aug_labels_dir = os.path.join(local_dataset_dir, "train_augmented.csv")
figures_dir = os.path.join(base_dir, "Figures")


images_dir = os.path.join(dataset_dir, "images")
images_augmented_dir = os.path.join(dataset_dir, "images_augmented")

labels_dir = os.path.join(dataset_dir, "train.csv")
labels_augmented_dir = os.path.join(dataset_dir, "train_augmented.csv")

transform_pipeline = transforms.Compose([
    transforms.CenterCrop(1280),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=random.randint(0, 180)),
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE))
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
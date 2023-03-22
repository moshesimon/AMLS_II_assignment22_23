import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from models import get_model
from train import train
from evaluate import evaluate
from data_loaders import get_data_loaders
from process import process_data

#process_data()

train_loader, val_loader = get_data_loaders()

model = get_model()

train(model, train_loader, val_loader)

evaluate(model, val_loader)

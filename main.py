import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from modules import train, process, data_loaders, models

process.process_data()

train_loader, val_loader = data_loaders.get_data_loaders()

model = models.get_model(num_meta_features=2, model_name='resnet50')

train.train(model, train_loader, val_loader)


import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import wandb
from config import *

def train(model, train_loader, val_loader):

    wandb.init(project="AMLII", job_type="training")

    wandb.watch(model)
    
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-5)

    # train the model
    print("Training model...")
    n_epochs = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_loader)
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        running_accuracy = 0.0
        #scheduler.step(epoch)
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_loader):
            data_, target_ = data_.to(device), target_.to(device) # on GPU
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(data_)

            pred = torch.sigmoid(outputs)
            target = target_.unsqueeze(1).float()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            pred = pred > 0.5
            accuracy = (target == pred).sum().item() / target.size(0)
            running_accuracy += accuracy
            wandb.log({"loss": loss.item(), 'step': batch_idx + batch_idx*(epoch-1), 'accuracy': 100 * accuracy})
            if (batch_idx) % 3 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * running_accuracy / total_step)
        train_loss.append(running_loss / total_step)
        wandb.log({"train_loss": running_loss / total_step, "train_accuracy": 100 * accuracy, 'epoch': epoch})

        torch.save(model.state_dict(), f'cancer_classification_{epoch}.pt')

        print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * accuracy):.4f}')
        batch_loss = 0
        batch_accuracy = 0
        with torch.no_grad():
            model.eval()
            for data_t, target_t in (val_loader):
                data_t, target_t = data_t.to(device), target_t.to(device) # on GPU
                outputs_t = model(data_t)
                pred_t = torch.sigmoid(outputs_t)
                target_t = target_t.unsqueeze(1).float()
                loss_t = criterion(pred_t, target_t)
                batch_loss += loss_t.item()
                pred_t = pred_t > 0.5
                accuracy_t = (target_t == pred_t).sum().item() / target_t.size(0)
                batch_accuracy += accuracy_t
            val_acc.append(100 * batch_accuracy / len(val_loader))
            val_loss.append(batch_loss / len(val_loader))
            wandb.log({"val_loss": batch_loss / len(val_loader), "val_accuracy": 100 * batch_accuracy / len(val_loader), 'epoch': epoch})

            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {(batch_loss / len(val_loader)):.4f}, validation acc: {(100 * batch_accuracy / len(val_loader)):.4f}\n')
            # Saving the best weight 
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(model.state_dict(), 'cancer_classification_best.pt')
                print('Detected network improvement, saving current model')
        model.train()

    fig = plt.figure(figsize = (20, 10))
    plt.title("Train - Validation Accuracy")
    plt.plot(train_acc, label = 'train')
    plt.plot(val_acc, label = 'validation')
    plt.xlabel('num_epochs', fontsize = 12)
    plt.ylabel('accuracy', fontsize = 12)
    plt.legend(loc = 'best')

    # save the figure
    fig.savefig(os.path.join(figures_dir,'train_val_acc.png'))
    plt.close()

    fig = plt.figure(figsize = (20, 10))
    plt.title("Train - Validation Loss")
    plt.plot(train_loss, label = 'train')
    plt.plot(val_loss, label = 'validation')
    plt.xlabel('num_epochs', fontsize = 12)
    plt.ylabel('loss', fontsize = 12)
    plt.legend(loc = 'best')
    # save the figure
    fig.savefig(os.path.join(figures_dir,'train_val_loss.png'))
    plt.close()

    wandb.finish()
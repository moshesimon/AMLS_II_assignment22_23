import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from config import *

def get_lr(optimizer):
    """Get the current learning rate of the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def log_metrics(train_loss, train_acc, val_loss, val_acc, pf_score, epoch, optimizer, all_targets, all_preds):
    """Log metrics to wandb and save confusion matrix."""
    # Log metrics to wandb
    wandb.log({
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1_score": pf_score,
        "epoch": epoch,
        "learning_rate": get_lr(optimizer)
    })

    # Save confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    plt.savefig(os.path.join(figures_dir, f'confusion_matrix_{epoch}.png'))
    plt.close()


def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def train(model, train_loader, val_loader):
    """Train the model and log metrics."""
    wandb.init(project="AMLII", job_type="training")
    wandb.watch(model)
    
    model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    n_epochs = 10
    max_f1_score = 0
    val_loss_hist, val_acc_hist, train_loss_hist, train_acc_hist = [], [], [], []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = 0, 0
        model.train()
        print(f"Epoch {epoch} of {n_epochs}")
        for batch_idx, (data_, metadata_, target_) in enumerate(train_loader):
            print(f"Batch {batch_idx} of {len(train_loader)}", end="\r")
            data_, metadata_, target_ = data_.to(device), metadata_.to(device), target_.to(device)
            optimizer.zero_grad()
            outputs = model(data_, metadata_)
            pred = torch.sigmoid(outputs)
            target = target_.unsqueeze(1).float()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            acc = (target == (pred > 0.5)).sum().item() / target.size(0)
            wandb.log({"loss": loss.item(), "accuracy": acc})
            train_acc += acc

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        val_loss, val_acc, all_preds, all_targets, all_logits = 0, 0, [], [], []
        model.eval()

        with torch.no_grad():
            for data_t, metadata_t, target_t in (val_loader):
                data_t, metadata_t, target_t = data_t.to(device), metadata_t.to(device), target_t.to(device)
                outputs_t = model(data_t, metadata_t)
                pred_t = torch.sigmoid(outputs_t)
                target_t = target_t.unsqueeze(1).float()
                loss_t = criterion(pred_t, target_t)

                val_loss += loss_t.item()
                val_acc += (target_t == (pred_t > 0.5)).sum().item() / target_t.size(0)
               
            all_preds.extend(pred_t.cpu().numpy().tolist())
            all_targets.extend(target_t.cpu().numpy().tolist())
            all_logits.extend(outputs.cpu().numpy().tolist())

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    # Convert logits to probabilities
    all_probabilities = [torch.sigmoid(torch.tensor(logit)).item() for logit in all_logits]

    pf_score = pfbeta(all_targets, all_probabilities, 1)

    log_metrics(train_loss, train_acc, val_loss, val_acc, pf_score, epoch, optimizer, all_targets, all_preds)
    scheduler.step(val_loss)

    torch.save(model.state_dict(), f'cancer_classification_{epoch}.pt')

    if pfbeta(all_targets, all_preds,) > max_f1_score:
        max_f1_score = f1_score(all_targets, all_preds)
        torch.save(model.state_dict(), 'cancer_classification_best.pt')

wandb.finish()

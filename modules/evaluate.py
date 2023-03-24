import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import wandb
from config import *

def evaluate(model, val_loader):
    wandb.init(project="AMLII", job_type="evaluation")
    # Importing trained Network with better loss of validation
    print("Loading model...")
    model.load_state_dict(torch.load('/scratch/zceemsi/AMLS_II_assignment22_23/weights/pious-valley-26/cancer_classification_5.pt'))

    # test the model
    print("Testing model...")

    model.to(device)
    model.eval()

    correct = 0
    total = 0   
    all_labels = []
    all_logits = []
    all_predicted = []

    with torch.no_grad():
        for data in val_loader:
            images, metadata, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = model(images, metadata)
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            
            # Calculate per-sample accuracy for binary classification
            correct += (predicted.squeeze() == labels).sum().item()
            
            # Store ground truth labels and predictions for the confusion matrix
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predicted.extend(predicted.cpu().numpy().tolist())
            all_logits.extend(outputs.cpu().numpy().tolist())

    # Flatten the ground truth labels and predictions
    flat_labels = np.array(all_labels).flatten()
    flat_predicted = np.array(all_predicted).flatten()

    # Calculate the confusion matrix
    cm = confusion_matrix(flat_labels, flat_predicted)

    # Display the confusion matrix
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    # Save the confusion matrix image
    plt.savefig(os.path.join(figures_dir,'confusion_matrix.png'))
    # Log the confusion matrix image to Weights & Biases
    wandb.log({"confusion_matrix": wandb.Image(plt)})

    accuracy = 100 * correct / total
    wandb.log({"val_accuracy": accuracy})
    print('Accuracy of the network on the validation set: %d %%' % (accuracy))

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
        
    # Convert logits to probabilities
    all_probabilities = [torch.sigmoid(torch.tensor(logit)).item() for logit in all_logits]

    # Calculate the probabilistic F1 score
    beta = 1
    pf1_score = pfbeta(all_labels, all_probabilities, beta)
    print('Probabilistic F1 score (pF1):', pf1_score)

    wandb.log({"pF1_score": pf1_score})

    wandb.finish()
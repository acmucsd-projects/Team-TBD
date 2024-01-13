import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# function to predict accuracy
def acc(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

def runEpochs(model, batch_size, train_loader, criterion, optimizer, valid_loader, device, modelDirec='models/state_dict.pt'):
    clip = 5
    epochs = 5
    valid_loss_min = np.Inf
    # train for some number of epochs
    epoch_tr_loss, epoch_vl_loss = [], []
    epoch_tr_acc, epoch_vl_acc = [], []

    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        model.train()
        # initialize hidden srate
        h = model.init_hidden(batch_size, device)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            train_losses.append(loss.item())
            # calculate accuracy
            accuracy = acc(output, labels)
            train_acc += accuracy
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs/LSTMs
            nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()

        val_h = model.init_hidden(batch_size, device)
        val_losses = []
        val_acc = 0.0
        model.eval()
        with torch.inference_mode():
            for input, labels in valid_loader:
                val_h = tuple([each.data for each in val_h])
                inputs, labels = inputs.to(device), labels.to(device)

                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

                accuracy = acc(output, labels)
                val_acc += accuracy

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc / len(train_loader.dataset)
        epoch_val_acc = val_acc / len(valid_loader.dataset)
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        print(f"Epoch {epoch + 1}")
        print(f"train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}")
        print(f"train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc}")
        if epoch_val_loss <= valid_loss_min:
            torch.save(model.state_dict(), modelDirec)
            print("Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...".format(valid_loss_min, epoch_val_loss))
            valid_loss_min = epoch_val_loss
        print(25*"==")
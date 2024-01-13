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

# Batching and loading as tensor
# Create Tensor datasets
def convertTensor(X_train_pad, X_test_pad, y_train, y_test, batch_size):
    train_data = TensorDataset(torch.from_numpy(X_train_pad), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(X_test_pad), torch.from_numpy(y_test))

    # dataloaders
    batch_size = 50

    # make sure to SHUFFLE data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader, valid_loader
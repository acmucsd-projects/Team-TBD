import dataprocessing as dp
import tokenizer
import padding as pad
import datasetTensors as dt
import model
import train_model as tm
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


if __name__ == "__main__":

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device('cuda')
        print("GPU is available")
    else:
        device = torch.device('cpu')
        print("GPU not available, CPU used")

    file_path = "input/mbti_1.csv"

    # load the data
    df = dp.load_data(file_path)

    # process the data
    df = dp.process(df, remove_special=True)
    df = dp.minWordBenchmark(df)

    # Train test split
    X, y = df['posts'].values, df['type'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
   
    # Tokenize
    X_train, y_train, X_test, y_test, vocab = tokenizer.tokenize(X_train, y_train, X_test, y_test)


    X_train_pad = pad.padding_(X_train, 500)
    X_test_pad = pad.padding_(X_test, 500)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # convert to tensors
    batch_size = 50
    train_loader, valid_loader = dt.convertTensor(X_train_pad, X_test_pad, y_train, y_test, batch_size)

    no_layers = 2
    vocab_size = len(vocab) + 1 # extra 1 for padding
    embedding_dim = 80
    output_dim = 1
    hidden_dim = 256

    # model
    model = model.MBTIRNN(no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5)

    model.to(device)

    print(model)

    lr = 0.005

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tm.runEpochs(model, batch_size, train_loader, criterion, optimizer, valid_loader, device)
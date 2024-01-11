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

def tokenize(X_train, y_train, X_val, y_val):
    word_list = []
    stop_words = set(stopwords.words('english'))
    labels = ['INFP' ,'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP' ,
              'ISFP' ,'ENTJ', 'ISTJ','ENFJ', 'ISFJ' ,'ESTP', 'ESFP' ,
              'ESFJ' ,'ESTJ']
    for sent in X_train:
        for word in sent.lower().split():
            # word = process_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w: i+1 for i,w in enumerate(corpus_)}

    # tockenize
    final_list_train, final_list_test = [], []
    for sent in X_train:
        final_list_train.append([onehot_dict[word] for word in sent.lower().split()
                                if word in onehot_dict.keys()])

    for sent in X_val:
        final_list_test.append([onehot_dict[word] for word in sent.lower().split()
                                if word in onehot_dict.keys()])
    encoded_train = [labels.index(label) for label in y_train]
    encoded_test = [labels.index(label) for label in y_val]
    return np.array(final_list_train,dtype='O'), np.array(encoded_train,dtype='O'), np.array(final_list_test,dtype='O'), np.array(encoded_test,dtype='O'), onehot_dict
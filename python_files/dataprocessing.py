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

def load_data(file_path="input/mbti_1.csv"):
    # Load the dataset from a CSV file
    df = pd.read_csv(file_path)
    return df

def process(df, remove_special=True):
    # Change to lowercase
    df['posts'] = df['posts'].apply(lambda x: x.lower())

    #Change case => lowercase
    df["posts"] = df["posts"].apply(lambda x: x.lower())

    #Remove acronyms of personality types within text, for accrate prediction with unknown data
    if remove_special:
        pers_types = ['INFP' ,'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP' ,'ISFP' ,'ENTJ', 'ISTJ','ENFJ', 'ISFJ' ,'ESTP', 'ESFP' ,'ESFJ' ,'ESTJ']
        pers_types = [p.lower() for p in pers_types]
        p = re.compile("(" + "|".join(pers_types) + ")")

    #Substitute hyperlinks with space
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'https?:\/\/.*?[\s+]', '', x.replace("|"," ") + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'https', '', x.replace("|"," ") + " "))

    # Substitute punctuations except EOS characters
        #Substitute all punctuation except EOS characters
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[\.+]', ".",x))  #remove punctuation
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^\w\s]','',x))  #avoid multiple full stops

    #Remove Numeric + Spl chars
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^a-zA-Z\s]','',x))

    #Remove multiple letters
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'([a-z])\1{2,}[\s|\w]*','',x))

    #Keep words within acceptable range (min letter 3, max 30)
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{0,3})?\b','',x))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{30,1000})?\b','',x))
    return df

def minWordBenchmark(df, min=10):
    df['Words'] = df['posts'].apply(lambda x: len(re.findall(r'\w+',x)))
    df = df[df['Words'] >= min]
    return df


# df = load_data()
# df = process(df, remove_special=True)
# df = minWordBenchmark(df)
# print(df.shape[0])


"""
ACM AI Project Team: TBD
This file contains function calls to train the model.
"""
# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel
import re

# Import utility functions
from util import *
from tokenization import *
from model import *
from train_model import *


if __name__ == "__main__":

    file_path = "../input/mbti_1.csv"

    # load the data
    df = pd.read_csv(file_path)

    # process the data
    df.process_df(df, remove_special=True)
    
    # Load BERT tokenizer
    tokenizer = get_tokenizer()
    
    # Mask input
    X_input_ids = np.zeros((len(df), 256))
    X_attn_masks = np.zeros((len(df), 256))
    
    # Generate embedded input data
    X_input_ids, X_attn_masks = tokenize_input(df, X_input_ids, X_attn_masks, tokenizer)
    
    # Encode MBTI labels
    labels = np.zeros((len(df), 16))
    labels[np.arange(len(df)), df['label'].values] = 1
    
    # Generate and map dataset for training
    dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks, labels))
    dataset = dataset.map(MBTIDatasetMapFunction)
    
    # Shuffle the dataset and generate batches, each with 16 training examples
    dataset = dataset.shuffle(10000).batch(16, drop_remainder=True)

    # Use 80/20 train-validation split
    p = 0.8
    batch_size = 16
    train_size = int((len(df)//batch_size)*p)

    # Generate train and validation data
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # Obtain the pretrained BERT model
    bert_model = load_BERT()

    # Obtain the customized model
    model = generate_model(bert_model)

    # Get optimizer, loss function, and accuracy metric
    learning_rate = 5e-5
    decay = 1e-6
    optim = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5, decay=1e-6)
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

    # Compile the model
    model.compile(optimizer=optim, loss=loss_func, metrics=[acc])

    # Train the model
    hist = model_train(model, train_dataset, val_dataset, epoch=17)

    # Save the trained model
    save_model(model, "MBTI_model")
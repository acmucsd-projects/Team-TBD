"""
ACM AI Project Team: TBD
This file contains the model we use for the MBTI classification task.
"""
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel
import re

def load_BERT():
    """
    Load the pretrained BERT model.
    """
    return TFBertModel.from_pretrained('bert-base-cased')

def generate_model(pretrained_model):
    """
    Customize the model for MBTI classification task using BERT model.
    """
    input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype='int32')
    attention_masks = tf.keras.layers.Input(shape=(256,), name='attention_mask', dtype='int32')

    bert_embds = pretrained_model.bert(input_ids, attention_mask=attention_masks)[1]
    intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name='intermediate_layer')(bert_embds)
    output_layer = tf.keras.layers.Dense(16, activation='softmax', name='output_layer')(intermediate_layer)

    model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output_layer)
    return model
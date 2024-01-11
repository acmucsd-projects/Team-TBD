import streamlit as st
import pandas as pd
import numpy as np
# from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
# from transformers import AutoTokenizer
import re

TYPES = ['ENFJ',
 'INTJ',
 'ENTP',
 'ESFP',
 'ESFJ',
 'ISTP',
 'INFJ',
 'ISFJ',
 'ENFP',
 'ISTJ',
 'ESTJ',
 'INTP',
 'ENTJ',
 'ISFP',
 'INFP',
 'ESTP']

st.title("What's your MBTI?")

# Utility function, need to change
def process_input(input_text, tokenizer, remove_special=True):
  # Change to lowercase
  input_text = input_text.lower()

  #Remove acronyms of personality types within text, for accrate prediction with unknown data
  if remove_special:
      pers_types = ['INFP' ,'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP' ,'ISFP' ,'ENTJ', 'ISTJ','ENFJ', 'ISFJ' ,'ESTP', 'ESFP' ,'ESFJ' ,'ESTJ']
      pers_types = [p.lower() for p in pers_types]
      p = re.compile("(" + "|".join(pers_types) + ")")

  #Substitute hyperlinks with space
  re.sub(r'https?:\/\/.*?[\s+]', '', input_text.replace("|"," ") + " ")
  re.sub(r'https', '', input_text.replace("|"," ") + " ")

  # Substitute punctuations except EOS characters
      #Substitute all punctuation except EOS characters
  re.sub(r'\.', ' EOSTokenDot ', input_text + " ")
  re.sub(r'\?', ' EOSTokenQuest ', input_text + " ")
  re.sub(r'!', ' EOSTokenExs ', input_text + " ")
  re.sub(r'[\.+]', ".",input_text)  #remove punctuation
  re.sub(r'[^\w\s]','',input_text)  #avoid multiple full stops

  #Remove Numeric + Spl chars
  re.sub(r'[^a-zA-Z\s]','',input_text)

  #Remove multiple letters
  re.sub(r'([a-z])\1{2,}[\s|\w]*','',input_text)

  #Keep words within acceptable range (min letter 3, max 30)
  re.sub(r'(\b\w{0,3})?\b','',input_text)
  re.sub(r'(\b\w{30,1000})?\b','',input_text)
  token = tokenizer.encode_plus(
      input_text,
      max_length=256,
      truncation=True,
      padding='max_length',
      add_special_tokens=True,
      return_tensors='tf'
  )
  return {
      'input_ids': tf.cast(token.input_ids, tf.float64),
      'attention_mask': tf.cast(token.attention_mask, tf.float64)
  }


# Load model
loaded_model = tf.keras.models.load_model('MBTI_model_6')

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Obtain input text
input_text = st.text_input('Some paragraphs that you personally wrote', 'Your input')

# Get tokenizer
tokenizer = tokenized_input_text = process_input(input_text, tokenizer)

# Obtain probabilities for each class
probs = loaded_model.predict(tokenized_input_text)

# MBTI Prediction
pred_index = np.argmax(probs[0])
pred_mbti = TYPES[pred_index]

# Second likely type
second_pred_index = np.argsort(np.max(probs, axis=0))[-2]
second_pred_mbti = TYPES[second_pred_index]

st.write('Your most likely MBTI type is', pred_mbti)

st.write('Your second most likely MBTI type is', second_pred_mbti)
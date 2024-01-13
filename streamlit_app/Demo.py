import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from transformers import BertTokenizer
import re
from dataStructures import *
from util import *

# Read in csv file
df = pd.read_csv("mbti_1.csv")

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.sidebar.success("Select a module above.")

# Title
st.title("Team TBD -- ACM AI Project 👋")

# Display author information
author_info = '''
Mentor: Vincent Tu

Team Members
* Hargen Zheng
* Catherine Zhang
* Phillip Wu
* Sia Patodia
* Ryan Wong
* Aryaman Dayal
'''
st.markdown(author_info)

# Load model
loaded_model = tf.keras.models.load_model('MBTI_model_6')

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

st.subheader("What's your MBTI?")

st.info('Thanks for using our Model to test your MBTI Personality!', icon="🔱")

# Obtain input text
input_text = st.text_area(
    "Text to analyze",
    "",
    )

if input_text != "":
    with st.spinner('Your Predicted Personality is TBD...'):
        time.sleep(5)
        # Get tokenizer
        tokenized_input_text = process_input(input_text, tokenizer)

        # Obtain probabilities for each class
        probs = loaded_model.predict(tokenized_input_text)

        # MBTI Predictions
        pred_index = np.argmax(probs[0])
        pred_mbti = MBTI_TYPES[pred_index]

        # Second likely type
        second_pred_index = np.argsort(np.max(probs, axis=0))[-2]
        second_pred_mbti = MBTI_TYPES[second_pred_index]

        # third likely type
        third_pred_index = np.argsort(np.max(probs, axis=0))[-3]
        third_pred_mbti = MBTI_TYPES[third_pred_index]
    st.success('Prediction has finished!')
    st.write('Based on the input, your most likely MBTI type is', pred_mbti)
    st.write('Based on the input, your second most likely MBTI type is', second_pred_mbti)
    st.write('Based on the input, your third most likely MBTI type is', third_pred_mbti)
    
    tab1, tab2, tab3 = st.tabs([pred_mbti, second_pred_mbti, third_pred_mbti])
    with tab1:
        st.write(MBTI_DESCRIPTIONS[pred_mbti])
    with tab2:
        st.write(MBTI_DESCRIPTIONS[second_pred_mbti])
    with tab3:
        st.write(MBTI_DESCRIPTIONS[third_pred_mbti])
    st.write("Credit to Myers & Briggs Foundation and 16personalities.com")
"""
ACM AI Project Team: TBD
This file contains utility functions needed to clean and process data.
Some other useful functions are also contained to deploy the streamlit app.
"""
import streamlit as st
import pandas as pd
import re
import tensorflow as tf

def mbti_dist_plot(df):
    """
    Plot the barchart that shows the distribution of MBTI personality
    types in the dataset we work with.
    """
    df_counts = df['type'].value_counts()
    count_df = pd.DataFrame(df_counts.index,df_counts.values).reset_index()
    count_df.columns = ['count', 'type']
    st.bar_chart(data=count_df, x='type', y='count')

# Utility function, need to change
def process_input(input_text, tokenizer, remove_special=True):
    """
    Utility function to clean and process the input text from the user.
    """
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
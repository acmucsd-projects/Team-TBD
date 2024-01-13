import streamlit as st
from dataStructures import *
from util import *

st.set_page_config(
    page_title="EDA",
    page_icon="👌",
)

st.write("# Welcome to our project! 👋")

st.sidebar.header("EDA Process")

st.markdown("# EDA Process")

st.subheader("See the distribution of MBTI types in our dataset.")

def mbti_dist_plot(df):
    """
    Plot the barchart that shows the distribution of MBTI personality
    types in the dataset we work with.
    """
    df_counts = df['type'].value_counts()
    count_df = pd.DataFrame(df_counts.index,df_counts.values).reset_index()
    count_df.columns = ['count', 'type']
    st.bar_chart(data=count_df, x='type', y='count')

# Read in csv file
df = pd.read_csv("mbti_1.csv")

# Plot the MBTI type distribution
mbti_dist_plot(df)

st.markdown(
    """
    - We needed to find a dataset which suited our goals the best for predicting MBTI personalities and we opted for this [dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type/data).
    - :orange[We assigned each of the 16 MBTI personality types a number from 0 - 15] to so that we could identify which personality was being predicted. In order to start training the model, we needed to do some data preprocessing steps. We removed all the special characters and numbers along with links. We only kept the words that were inside the allowed range of 3 - 30 characters. We then used the pretrained BERT tokenizer in order to tokenize the posts in the dataset. 
        
"""
)
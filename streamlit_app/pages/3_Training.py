import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Training Process", page_icon="📈")

st.markdown("# Training Process")
st.sidebar.header("Training")

content_1 = """
We trained the model by giving it a fraction of the dataset and masking some of the words and making it try to predict the missing words in order to get the personality of the post. \
The model then assigned a personality to the post and then it was checked with the original dataset to see if it was correct.

## Model Input
With BERT model's pretained tokenizer, we are able to embed each text corpus into a :orange[feature vector with the maximum \
length of 256], thus feeding into the network and train the model. Since the task BERT was trained on was quite \
different from our task -- to predict a person's personality based on text corpus -- we decided to train all the parameters with \
Google Colab's GPU, in hope to achieve a reasonably good performance. 

## Train-Validation Split and Batch Training
To evaluate our model performance on unseen data, we used :orange[$80-20$ split into training and validation dataset], dropping the remainders. \
As we have $8675$ unique entries in the dataset, this results in :orange[$6928$ examples in our training dataset]. We used the :orange[batch size of $16$], meaning \
we have $433$ batches for each epoch.


## Learning Rate 
To avoid overshooting when the model does backpropagation, \
we initially used a standard learning rate for training transformer models: $1e^{-5}$. Then, it turns out we did not get much progress \
after 5 epochs in 30 minutes. Therefore, we tried $1e
^{-4}$ to see if model learns better, but the gradient seemed to explode after the first epoch, \
resulting in a continued drop in accuracy. Then, we halved the ambitious :orange[learning rate to $5e^{-5}$ with decay of $1e^{-6}$ using the Adam optimizer and \
the model starts to learn at a reasonable speed]. Eventually, we obtained the ideal model after $17$ epochs. :orange[The training accuracy \
goes from $26.57\%$ after the first epoch to $94.79\%$ after the $17$th epoch], which is a huge progress!

## Training History
"""
st.markdown(content_1)

st.image('./pages/train_1.png')
st.image('./pages/train_2.png')
st.image('./pages/train_3.png')
st.image('./pages/train_4.png')
st.image('./pages/train_5.png')
st.image('./pages/train_6.png')
st.image('./pages/train_7.png')

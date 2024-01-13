import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="BERT Model", page_icon="📈")

st.markdown("# BERT Model")
st.sidebar.header("Model")

content = """
Since the size of our data set is relatively small, we decided to use transfer learning based on \
the :orange[BERT (Bidirectional Encoder Representations from Transformers)] model. This way, we do not have a \
code start issue where we would computational time and power to get close to the minimum of the objective function.
"""
st.markdown(content)

st.image('./pages/BERT.png', caption="BERT Architecture")

content_2 = """
The model is bidirectional, meaning that it can read the text from left to right and also right to left, thus capturing \
more nuances within the input text corpus. Similar to the standard transformer model, BERT has an :orange[encoding network, which \
embeds the input text corpus in a way that close meaning words are closer together in the high dimensional feature representation \
whereas less related words are incentivized to stay farther apart during the training process]. This way, we can capture the semantics \
within the text corpus and better understand the text.

Then, the output of the encoder network is fed into the :orange[decoder network] to try to extract the features within the \
feature representation. With layers of convolution, followed by some fully connected layers, the decoder network would be ready to \
go through the :orange[softmax layer for multi-class classification task]. In our case, since we are predicting one of $16$ personalities, we have \
$16$ units in the softmax layer, thus outputting a 16-dimensional vector representing the probabilities of each MBTI personality type.
"""
st.markdown(content_2)

st.image('./pages/Softmax.png', caption='Softmax Layer')
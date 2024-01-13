"""
ACM AI Project Team: TBD
This file contains utility functions to make predictions using our model.
"""
import numpy as np

from util import *
from data_structures import *

def tokenize_text(input_text, tokenizer):
    return process_input(input_text, tokenizer)

def predict_most_likely(model, tokenized_input_text):
    probs = model.predict(tokenized_input_text)
    pred_index = np.argmax(probs[0])
    pred_mbti = MBTI_TYPES[np.argmax(probs[0])]
    return pred_mbti
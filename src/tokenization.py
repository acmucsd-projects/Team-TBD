"""
ACM AI Project Team: TBD
This file contains utility functions needed to tokenize the input text.
"""
from transformers import BertTokenizer

def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-cased')

def tokenize_input(df, ids, masks, tokenizer):
    """"
    Helps embed the input text into a feature vector of maximum length of 256.
    """
    for i, text in tqdm(enumerate(df['posts'])):
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=256,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks
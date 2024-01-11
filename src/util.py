"""
This file contains utility functions that are helpful for
data processing for the MBTI classification project.
"""

from collections import defaultdict
import string
from nltk.stem.porter import *

def get_wordCounts(data):
    wordCount = defaultdict(int)
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    for type, post in data:
        r = ''.join(c for c in post.lower() if not c in punctuation)
        for w in r.split():
            w = stemmer.stem(w)
            wordCount[w] += 1
    counts = [(wordCount[w], w) for w in wordCount]
    counts.sort()
    counts.reverse()
    return counts

# def convert_text(data, words)
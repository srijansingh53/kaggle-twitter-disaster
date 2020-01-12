import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas_profiling as pp
import re
import string
from autocorrect import spell
import nltk
from nltk.stem import WordNetLemmatizer 

class Process():

    def preprocess_text(self, text):
        
        # Removing links- http
        text =  re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    #    # Removing mentions and hashtags
    #    text = re.sub(r"#(\w+)", ' ', text, flags=re.MULTILINE)
    #    text = re.sub(r"@(\w+)", '', text, flags=re.MULTILINE)
        # Removing punctuations
        text = re.sub(r'[^\w\d\s]', '', text)
        # convert to lower case
        text = re.sub(r'^\s+|\s+?$', '', text.lower())
        # Removing digits
        text = re.sub(r'\d', '', text)
        # Removing other symbols
        text = re.sub('[ãâª]+', '', text)
        # collapse all white spaces
        text = re.sub(r'\s+', ' ', text)
        # remove stop words and perform stemming
        stop_words = nltk.corpus.stopwords.words('english')
        # 
        lemmatizer = WordNetLemmatizer() 
        return ' '.join(
            lemmatizer.lemmatize(term) 
            for term in text.split()
            if term not in set(stop_words)
        )


#from textblob import Word,TextBlob
#w = Word('falibility')
#w.spellcheck()
#
#b = TextBlob("disinfo")
#print(b.correct())






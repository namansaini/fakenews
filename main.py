# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 
from gensim import corpora, models, similarities
import itertools
import gensim
def clean_text(text):
    #Remove URLs
    #text = re.sub(r"http\S+", "", str(text))
    #Tokenize
    text = text.lower()
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    tokens = tokenizer.tokenize(text)
    #Remove non alphanumerica characters
    stop_words = set(stopwords.words('english'))
    words=[word for word in tokens if not word in stop_words] 
    words = [word for word in tokens if word.isalpha()]    
    return words

data = pd.read_csv("fake.csv", usecols=[2,4,5,8])
data.head()
data.isnull().sum()

#removing null values
mData = data.dropna().loc[:]
mData.head()
mData.isnull().sum() 

mData['author'] = mData.apply(lambda row: clean_text(row['author']), axis=1)
mData['title'] = mData.apply(lambda row: clean_text(row['title']), axis=1)
mData['text'] = mData.apply(lambda row: clean_text(row['text']), axis=1)
#mData['site_url'] = mData.apply(lambda row: clean_text(row['site_url']), axis=1)
mData.head()
res_list = list(itertools.chain(*mData['title'].values.tolist()))
model = gensim.models.Word2Vec([res_list], min_count=1, size = 32)
model.wv['busted']
model.train([res_list], total_examples=1, epochs=1)
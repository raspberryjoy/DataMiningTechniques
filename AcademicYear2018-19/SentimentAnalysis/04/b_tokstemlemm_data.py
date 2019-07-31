#!/usr/bin/env python
# -*- coding: utf-8 -*-



################ IMPORTS

import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer

from aux import *


   
    
    
## Tokenization & stemming

def token_stem_lemm():
    
    # Load the pre-processed data
    train_df = pickle.load(open(train_df_prep_filename, "rb"))


    # Porter stemmer
    porter = PorterStemmer() 
    # Lemmatizer
    lemmatizer = WordNetLemmatizer()

    
    tokstem_tweets = train_df['tweet'].apply(lambda x: ' '.join([porter.stem(item) for item in word_tokenize(x)]))
    toklemm_tweets = train_df['tweet'].apply(lambda x: ' '.join([lemmatizer.lemmatize(item) for item in word_tokenize(x)]))    

    train_df = []

    # Save the words
    pickle.dump(tokstem_tweets, open(tokstem_tweets_filename, "wb"))
    pickle.dump(toklemm_tweets, open(toklemm_tweets_filename, "wb")) 
    

    #
    test_df = pickle.load(open(test_df_prep_filename, "rb"))

    tokstem_tweets = test_df['tweet'].apply(lambda x: ' '.join([porter.stem(item) for item in word_tokenize(x)]))
    toklemm_tweets = test_df['tweet'].apply(lambda x: ' '.join([lemmatizer.lemmatize(item) for item in word_tokenize(x)]))    
    

    # Save the words
    pickle.dump(tokstem_tweets, open(tokstem_tweets_test_filename, "wb"))
    pickle.dump(toklemm_tweets, open(toklemm_tweets_test_filename, "wb"))  





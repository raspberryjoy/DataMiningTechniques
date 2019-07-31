#!/usr/bin/env python
# -*- coding: utf-8 -*-


################ IMPORTS

import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle
from nltk.corpus import stopwords

from aux import *





################ FUNCTIONS

def prepSave_data(data_filename, stop_words, data_df_prep_filename):

    # Load the data
    data_df = pd.read_csv(data_filename, 
                    sep="\t", index_col=False, encoding='utf-8',
                    names=['id1', 'id2', 'class', 'tweet'])


    # Process each tweet
    
    # lowercase
    data_df.tweet = data_df.tweet.str.lower()    
    # rm unicode characters
    data_df.tweet = data_df.tweet.str.replace(r"\\u[a-zA-Z0-9]{4}", "")
    # rm mentions, hashtags, links
    data_df.tweet = data_df.tweet.str.replace(r"#\S+", "")
    data_df.tweet = data_df.tweet.str.replace(r"@\S+", "")
    data_df.tweet = data_df.tweet.str.replace(r"http\S+", "")
    # rm emoticons (source: https://stackoverflow.com/questions/28077049/regex-matching-emoticons) 
    data_df.tweet = data_df.tweet.str.replace(r"(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)", "") 
    # rm numbers, punctuation, symbols 
    data_df.tweet = data_df.tweet.str.replace(r"[0-9.$%!=+'-\\&]+", "")
    # rm stopwords
    data_df.tweet = data_df.tweet.apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words])) 
    # rm multiple consecutive spaces   
    data_df.tweet = data_df.tweet.str.replace(r"\s{2,}", " ")
    # stripping
    data_df.tweet = data_df.tweet.str.strip()

        
    # Save the pre-processed dataframe 
    pickle.dump(data_df, open(data_df_prep_filename, "wb"))

################ END FUNCTIONS





def prep_data():
    
    ## Pre-processing of the training and test data
    # Filter: metions, hashtags, links, emoticons, stopwords, punctuation, symbols, numbers
    # lowercase, stripping
        
    # Cache the stop-words to speed up the process
    stop_words = set(stopwords.words('english'))

    extra_words = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                    'mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun',
                    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
                    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov', 'dec',
                    'rt', 'im', 'today', 'tonight', 'yesterday', 'tomorrow', 'night', 'morning', 'day']

    stop_words.update(extra_words)


    # Pre-process and save the data
    prepSave_data(train_data_filename, stop_words, train_df_prep_filename)
    prepSave_data(test_data_filename, stop_words, test_df_prep_filename)

    
                   


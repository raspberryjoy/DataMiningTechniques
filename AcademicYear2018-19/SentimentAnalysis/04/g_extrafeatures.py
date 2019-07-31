#!/usr/bin/env python
# -*- coding: utf-8 -*-



################ IMPORTS

import re
import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle
from random import randint
from nltk.tokenize import word_tokenize 

from aux import *
#from pca_funs import pca_mat





################ FUNCTIONS

def extrafeatures(data_filename, data_init_filename, extrafeatures_mat_filename, lexica, 
                    tweets_hashtags_file, pca_extrafeatures_mat_filename,
                    pca_dims=3, pca_whiten=True):  

    data_df = pickle.load(open(data_filename,"rb"))   
    #
    data_tweets = data_df['tweet']
    
    #
    data_init_df = pd.read_csv(data_init_filename, 
                    sep="\t", index_col=False, encoding='utf-8',
                    names=['id1', 'id2', 'class', 'tweet'])
      
   
    # length of tweet, mean / min. / max. valence per dictionary
    extrafeatures_mat = []  
    
    #
    extrafeatures_mat = np.array([[0 for x in range(idx['end'])] for y in range(len(data_tweets))])
    
    tweets_hashtags = pickle.load(open(tweets_hashtags_file, "rb"))
    
    top20_poshashtags = pickle.load(open(top20_hashtags_pos_filename, "rb"))
    top20_neghashtags = pickle.load(open(top20_hashtags_neg_filename, "rb"))
    top20_neuhashtags = pickle.load(open(top20_hashtags_neu_filename, "rb"))

    hash_counter = lambda tweet: sum([item in tweets_hashtags for item in tweet.split() if item in top20_poshashtags])
    vfun = np.vectorize(hash_counter)    
    extrafeatures_mat[:,idx['num_top_poshash']] = vfun(tweets_hashtags)
    
    hash_counter = lambda tweet: sum([item in tweets_hashtags for item in tweet.split() if item in top20_neghashtags])
    vfun = np.vectorize(hash_counter)    
    extrafeatures_mat[:,idx['num_top_neghash']] = vfun(tweets_hashtags)
    
    hash_counter = lambda tweet: sum([item in tweets_hashtags for item in tweet.split() if item in top20_neuhashtags])
    vfun = np.vectorize(hash_counter)    
    extrafeatures_mat[:,idx['num_top_neuhash']] = vfun(tweets_hashtags)  
       
    tweets_hashtags = []; top20_poshashtags = []; top20_neghashtags = []; top20_neuhashtags = []     


    hash_counter = lambda tweet: sum([item in ['a','e','i','o','u','y'] for item in list(tweet)])
    vfun = np.vectorize(hash_counter)    
    extrafeatures_mat[:,idx['num_vowels']] = vfun(data_tweets)                  
            
            
          

    lexico_idx = -3 

    # run through each dictionary
    for lexico in lexica:

        # Set the idex of the current dictionary
        lexico_idx += 3

        # Read the dictionary
        lexico_dict = pd.read_csv(lexico, 
                                sep="\t", index_col=False, encoding='utf-8',
                                names=['word','value'])
        
        # Convert it to dict                        
        lexico_dict = lexico_dict.set_index('word').T.to_dict()
        
        
        # data
        for i in range(len(data_tweets)):         
            
            # No. words in the current dictionary
            num_found = 0
            
            # Save the min. / max. valence
            min_val = float('Inf')
            max_val = -float('Inf')
            
            # Tokenize the current tweet
            cur_tokens = word_tokenize(data_tweets[i])
            
            # Save the length of the tweet, no. words / emotions / exclamations marks / urls / hashtags / mentions
            extrafeatures_mat[i][idx['tweet_len']] = len(data_init_df.loc[i,'tweet'])
            extrafeatures_mat[i][idx['num_words']] = len(word_tokenize(data_init_df.loc[i,'tweet']))
            
            
            extrafeatures_mat[i][idx['num_emoticons']] = re.subn("(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)", '', data_init_df.loc[i,'tweet'])[1]
            extrafeatures_mat[i][idx['num_excl']] = re.subn("!", '', data_init_df.loc[i,'tweet'])[1]
            extrafeatures_mat[i][idx['num_urls']] = re.subn("http\S+", '', data_init_df.loc[i,'tweet'])[1]
            extrafeatures_mat[i][idx['num_hashtags']] = re.subn("#\S+", '', data_init_df.loc[i,'tweet'])[1]
            extrafeatures_mat[i][idx['num_mentions']] = re.subn("@\S+", '', data_init_df.loc[i,'tweet'])[1]            


            
            #Check if any token exists in the current dictionary
            for token in cur_tokens:
                if token in lexico_dict:
                
                    cur_valence = lexico_dict.get(token)['value']
                
                    num_found += 1
                    extrafeatures_mat[i][lexico_idx] += cur_valence
                    
                    # Check if the min. / max valence should be updated
                    if cur_valence < min_val:
                        min_val = cur_valence
                    if cur_valence > max_val:
                        max_val = cur_valence
            

            # Set the final values of the current mean / min. / max. valence
            if 0 == num_found:
                extrafeatures_mat[i][lexico_idx], min_val, max_val = random_mean_valence(lexico_dict)
            else:
                extrafeatures_mat[i][lexico_idx] /= num_found

            # min. / max. valence
            extrafeatures_mat[i][lexico_idx + 1] = min_val
            extrafeatures_mat[i][lexico_idx + 2] = max_val
        
        
    # Save the data
    pickle.dump(extrafeatures_mat, open(extrafeatures_mat_filename, "wb"))  

    extrafeatures_mat = []; data_tweets = []  
    
    
    # PCA
    #pca_mat(extrafeatures_mat_filename, pca_extrafeatures_mat_filename, pca_dims, pca_whiten)   
    
    
    
def random_mean_valence(lexico_dict):

    num_samples = 100
    rand_mean_valence = 0
    max_dict_idx = len(lexico_dict) - 1
    min_val = float('Inf')
    max_val = -float('Inf')
    
    
    for i in range(0, num_samples, 1):
    
        cur_valence = lexico_dict.get(lexico_dict.keys()[randint(0, max_dict_idx)])['value']
        
        rand_mean_valence += cur_valence
        
        if min_val > cur_valence:
            min_val = cur_valence
            
        if max_val < cur_valence:
            max_val = cur_valence


    return (rand_mean_valence / num_samples), min_val, max_val 

################ END FUNCTIONS





## Extra features

def extrafeat():
    
    
    extrafeatures(train_df_prep_filename, train_init_filename, extrafeatures_mat_filename, lexica, 
                    tweets_hashtags_filename, pca_extrafeatures_mat_filename)
                    
    extrafeatures(test_df_prep_filename, test_init_filename, extrafeatures_test_mat_filename, lexica, 
                    tweets_hashtags_test_filename, pca_extrafeatures_test_mat_filename)
    
    
   


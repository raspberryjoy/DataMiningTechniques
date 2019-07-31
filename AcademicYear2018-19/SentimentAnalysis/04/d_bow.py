#!/usr/bin/env python
# -*- coding: utf-8 -*-



################ IMPORTS

import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer

from aux import *
#from pca_funs import sparsepca_array, sparsepca_mat



    
    
  
    
## Bag-of-words

def bow(countvec_max_features=50, max_df=0.5, min_df=1, pca_dims=3):
      
    train_df = pickle.load(open(train_df_prep_filename,"rb"))   
    #
    train_tweets = train_df['tweet']


    # Set the list of text documents - tweets of the training data
    text = list(train_tweets)
           
    vectorizer = CountVectorizer(max_features=countvec_max_features,
                                max_df=max_df,
                                min_df=min_df)

    bow_mat = vectorizer.fit_transform(text)    


    # Save the bow matrix
    pickle.dump(bow_mat, open(bow_mat_filename, "wb"))

    train_tweets = []; bow_mat = []



    test_df = pickle.load(open(test_df_prep_filename,"rb"))   
    #
    test_tweets = test_df['tweet']    

    # Set the list of text documents - tweets of the test data
    text = list(test_tweets)

    bow_test_mat = vectorizer.transform(text)



    # Save the bow matrix
    pickle.dump(bow_test_mat, open(bow_test_mat_filename, "wb"))

    test_tweets = []; bow_test_mat = [];
        
    

    # PCA
    #sparsepca_array(bow_mat_filename, pca_bow_mat_filename, pca_dims)
    #sparsepca_array(bow_test_mat_filename, pca_bow_test_mat_filename, pca_dims)  
        
    
      
    

#!/usr/bin/env python
# -*- coding: utf-8 -*-



################ IMPORTS

import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from aux import *
#from pca_funs import pca_mat


   
   
    
    
## TF-IDF

def tfidf(tfidf_max_features=50, max_df=0.5, min_df=1, pca_dims=3, pca_whiten=True):
    
    #
    train_df = pickle.load(open(train_df_prep_filename,"rb"))   
    #
    train_tweets = train_df['tweet']
    
        
    # list of text documents
    text = list(train_tweets)
    
    # create the transform
    vectorizer = TfidfVectorizer(input='content',
                                max_df=max_df,
                                min_df=min_df,
                                max_features=tfidf_max_features, # build a vocabulary that only consider the top max_features ordered, ignored if vocabulary is not None
                                use_idf=True, # inverse-document-frequency reweighting
                                smooth_idf=True, # prevents zero divisions
                                sublinear_tf=False) # sublinear tf scaling, i.e. replace tf with 1 + log(tf)
    # Build vocabulary
    X = vectorizer.fit(text)    


    # Training data
    tfidf_mat = []    

    for i in range(len(train_tweets)):

        v = vectorizer.transform([text[i]])
        tfidf_mat.append(v.toarray()[0])


    # Save the bow matrix
    pickle.dump(tfidf_mat, open(tfidf_mat_filename, "wb"))


    tfidf_mat = []
        
   
   
   
    test_df = pickle.load(open(test_df_prep_filename,"rb"))   
    #
    test_tweets = test_df['tweet'] 
      
      
    text = list(test_tweets)                 
    
    tfidf_test_mat = []    

    for i in range(len(test_tweets)):
       
        v = vectorizer.transform([text[i]])
        tfidf_test_mat.append(v.toarray()[0])


    # Save the bow matrix
    pickle.dump(tfidf_test_mat, open(tfidf_test_mat_filename, "wb"))    


    tfidf_test_mat = []



    # PCA
    #pca_mat(tfidf_mat_filename, pca_tfidf_mat_filename, pca_dims, pca_whiten)
    #pca_mat(tfidf_test_mat_filename, pca_tfidf_test_mat_filename, pca_dims, pca_whiten)
    
    
 
 

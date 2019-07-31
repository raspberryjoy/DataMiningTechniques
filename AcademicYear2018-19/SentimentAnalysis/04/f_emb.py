#!/usr/bin/env python
# -*- coding: utf-8 -*-



################ IMPORTS

import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle
from operator import add
from random import randint
from nltk.tokenize import word_tokenize 
import gensim

from aux import *
#from pca_funs import pca_mat



################ FUNCTIONS

def random_mean_vec(model):
        
    num_samples = 100
    rand_vec = []


    for i in range(0, num_samples, 1):

        if len(rand_vec) == 0:
            rand_vec = model.wv[model.wv.vocab.keys()[randint(0, len(model.wv.vocab) - 1)]]

        else:
            rand_vec = np.add(rand_vec, model.wv[model.wv.vocab.keys()[randint(0, len(model.wv.vocab) - 1)]])


    rand_vec = np.divide(rand_vec, num_samples)

    return(rand_vec)

################ END FUNCTIONS



    
   
   
    
    
## Word emeddings

#
def emb(w2v_size=50, w2v_window=10, w2v_min_count=2, w2v_sg=0, epochs=10, pca_dims=3, pca_whiten=True):
    

    train_df = pickle.load(open(train_df_prep_filename,"rb"))   
    #
    train_tweets = train_df['tweet']

    text = [word_tokenize(lemm) for lemm in train_tweets]

    # Word2Vec
    model = gensim.models.Word2Vec(text,
                                    size=w2v_size,
                                    window=w2v_window,
                                    min_count=w2v_min_count,
                                    sg=w2v_sg) # CBOW (0) or skip gram (1)
        
    model.train(text, 
                total_examples=len(train_tweets),
                epochs=epochs)



    
    # Mean vectors - training data
    emb_mat = []    


    for lemm in train_tweets:
        
        num_found = 0
        cur_mean_vec = []
        
        cur_tokens = word_tokenize(lemm)
        
        
        for token in cur_tokens:
            if token in model.wv.vocab.keys():
            
                num_found += 1

                if len(cur_mean_vec) == 0:
                    cur_mean_vec = model.wv.word_vec(token)
                else:
                    cur_mean_vec = np.add(cur_mean_vec, model.wv.word_vec(token))
                        
        
        
        if 0 == num_found:
            cur_mean_vec = random_mean_vec(model)
        else:
            cur_mean_vec = np.divide(cur_mean_vec, num_found)
        
        # Add the final vector
        emb_mat.append(cur_mean_vec)   


    # Save the data
    if 0 == w2v_sg:
        pickle.dump(emb_mat, open(emb0_mat_filename, "wb"))
    else:
        pickle.dump(emb_mat, open(emb1_mat_filename, "wb"))

    emb_mat = []; train_tweets = []; text = []
        
 

 
 
    #
    test_df = pickle.load(open(test_df_prep_filename,"rb"))   
    #
    test_tweets = test_df['tweet']      
            
    # Mean vectors - test data  
    emb_test_mat = []


    for lemm in test_tweets:

        num_found = 0
        cur_mean_vec = []
        
        cur_tokens = word_tokenize(lemm)
        
        
        for token in cur_tokens:
            if token in model.wv.vocab.keys():
            
                num_found += 1

                if 0 == len(cur_mean_vec):
                    cur_mean_vec = model.wv.word_vec(token)
                else:
                    cur_mean_vec = np.add(cur_mean_vec, model.wv.word_vec(token))
                        
                
        if 0 == num_found:
            cur_mean_vec = random_mean_vec(model)
        else:
            cur_mean_vec = np.divide(cur_mean_vec, num_found)
        
        # Add the final vector
        emb_test_mat.append(cur_mean_vec) 
            
        
    # Save the data   
    if 0 == w2v_sg:
        pickle.dump(emb_test_mat, open(emb0_test_mat_filename, "wb"))
    else:         
        pickle.dump(emb_test_mat, open(emb1_test_mat_filename, "wb"))

    
    emb_test_mat = []; test_tweets = []
    
    
           
    # PCA
    """
    if 0 == w2v_sg:
        pca_mat(emb0_mat_filename, pca_emb0_mat_filename, pca_dims, pca_whiten)
        pca_mat(emb0_test_mat_filename, pca_emb0_test_mat_filename, pca_dims, pca_whiten)
                
    else:
        pca_mat(emb1_mat_filename, pca_emb1_mat_filename, pca_dims, pca_whiten)
        pca_mat(emb1_test_mat_filename, pca_emb1_test_mat_filename, pca_dims, pca_whiten) 
    """  
    
    
    
    
    
#
def embGoogle(pca_dims=3, pca_whiten=True, limit=500000):
    

    train_df = pickle.load(open(train_df_prep_filename,"rb"))   
    #
    train_tweets = train_df['tweet']


    model = gensim.models.KeyedVectors.load_word2vec_format("/home/user/Downloads/GoogleNews-vectors-negative300.bin", binary=True, limit=limit)



    
    # Mean vectors - training data
    emb_mat = []    


    for lemm in train_tweets:
        
        num_found = 0
        cur_mean_vec = []
        
        cur_tokens = word_tokenize(lemm)
        
        
        for token in cur_tokens:
            if token in model:
            
                num_found += 1

                if len(cur_mean_vec) == 0:
                    cur_mean_vec = model[token]
                else:
                    cur_mean_vec = np.add(cur_mean_vec, model[token])
                        
        
        
        if 0 == num_found:
            cur_mean_vec = random_mean_vec(model)
        else:
            cur_mean_vec = np.divide(cur_mean_vec, num_found)
        
        # Add the final vector
        emb_mat.append(cur_mean_vec)   


    # Save the data
    pickle.dump(emb_mat, open(emb0_mat_filename + ".google", "wb"))

    emb_mat = []; train_tweets = []; text = []
        
 

 
 
    #
    test_df = pickle.load(open(test_df_prep_filename,"rb"))   
    #
    test_tweets = test_df['tweet']      
            
    # Mean vectors - test data  
    emb_test_mat = []


    for lemm in test_tweets:

        num_found = 0
        cur_mean_vec = []
        
        cur_tokens = word_tokenize(lemm)
        
        
        for token in cur_tokens:
            if token in model:
            
                num_found += 1

                if 0 == len(cur_mean_vec):
                    cur_mean_vec = model[token]
                else:
                    cur_mean_vec = np.add(cur_mean_vec, model[token])
                        
                
        if 0 == num_found:
            cur_mean_vec = random_mean_vec(model)
        else:
            cur_mean_vec = np.divide(cur_mean_vec, num_found)
        
        # Add the final vector
        emb_test_mat.append(cur_mean_vec) 
            
        
    # Save the data   
    pickle.dump(emb_test_mat, open(emb0_test_mat_filename + ".google", "wb"))

    
    emb_test_mat = []; test_tweets = []        
    
    
    
   

#!/usr/bin/env python
# -*- coding: utf-8 -*-


################ IMPORTS

from os import path
import re, sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from aux import *



################ FUNCTIONS

def knn_class(data_file, data_test_file, y_test_filename, knn_n=3, knn_algorithm='auto'):           

    y_train = pickle.load(open(y_train_filename,"rb"))
                                    
    data_train = pickle.load(open(data_file, "rb"))
    data_test = pickle.load(open(data_test_file, "rb"))              
            
        
    # KNN classifier
    
    knn_classifier = KNeighborsClassifier(n_neighbors=knn_n, algorithm=knn_algorithm)                                 


    knn_classifier.fit(data_train, y_train)
    y_test = knn_classifier.predict(data_test)

    data_train = []; data_test = []

    #
    pickle.dump(y_test, open(y_test_filename, "wb"))
    

################ END FUNCTIONS    
    
    
    
    


## KNN

def knn(knn_n=3, knn_algorithm='auto'):
   
    if 1 == knn_n:
            
        knn_class(bow_mat_filename, bow_test_mat_filename, y_pred_knn1_bow_file, knn_n, knn_algorithm)    
        knn_class(tfidf_mat_filename, tfidf_test_mat_filename, y_pred_knn1_tfidf_file, knn_n, knn_algorithm)    
        knn_class(emb0_mat_filename, emb0_test_mat_filename, y_pred_knn1_ebmb0_file, knn_n, knn_algorithm)     
        knn_class(emb1_mat_filename, emb1_test_mat_filename, y_pred_knn1_emb1_file, knn_n, knn_algorithm)         
        knn_class(extrafeatures_mat_filename, extrafeatures_test_mat_filename, y_pred_knn1_extra_file, knn_n, knn_algorithm)     
        knn_class(embextra0_mat_filename, embextra0_test_mat_filename, y_pred_knn1_embextra0_file, knn_n, knn_algorithm)     
        knn_class(embextra1_mat_filename, embextra1_test_mat_filename, y_pred_knn1_embextra1_file, knn_n, knn_algorithm) 
                    
                
    elif 3 == knn_n:
        
        knn_class(bow_mat_filename, bow_test_mat_filename, y_pred_knn3_bow_file, knn_n, knn_algorithm)    
        knn_class(tfidf_mat_filename, tfidf_test_mat_filename, y_pred_knn3_tfidf_file, knn_n, knn_algorithm)    
        knn_class(emb0_mat_filename, emb0_test_mat_filename, y_pred_knn3_ebmb0_file, knn_n,  knn_algorithm)     
        knn_class(emb1_mat_filename, emb1_test_mat_filename, y_pred_knn3_emb1_file, knn_n, knn_algorithm)         
        knn_class(extrafeatures_mat_filename, extrafeatures_test_mat_filename, y_pred_knn3_extra_file, knn_n, knn_algorithm)     
        knn_class(embextra0_mat_filename, embextra0_test_mat_filename, y_pred_knn3_embextra0_file, knn_n, knn_algorithm)     
        knn_class(embextra1_mat_filename, embextra1_test_mat_filename, y_pred_knn3_embextra1_file, knn_n, knn_algorithm) 
                    





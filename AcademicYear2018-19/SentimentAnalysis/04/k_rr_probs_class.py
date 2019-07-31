#!/usr/bin/env python
# -*- coding: utf-8 -*-


################ IMPORTS

import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from aux import *



################ FUNCTIONS

def rr_probs_class(data_file, data_test_file, y_test_filename, knn_n=3, knn_algorithm='auto'):

    y_train = np.array(pickle.load(open(y_train_filename,"rb")))

                                  
    data_train = pickle.load(open(data_file, "rb"))
    data_test = pickle.load(open(data_test_file, "rb"))

    neg_rows = pickle.load(open(x_train_neg_filename,"rb"))
    neu_rows = pickle.load(open(x_train_neu_filename,"rb"))
    pos_rows = pickle.load(open(x_train_pos_filename,"rb"))

    NeuP_rows = np.array(neu_rows + pos_rows)
    NeuNeg_rows = np.array(neu_rows + neg_rows)
    PNeg_rows = np.array(neg_rows + pos_rows)
    
    neg_rows = []; neu_rows = []; pos_rows = []
    
        

    ############# Round-robin classification: probs

    if data_file == "gendata/bow_mat.pkl" or data_file == "gendata/bow_mat.pkl":
        y_train_All = np.array([[0 for x in range(6)] for y in range(data_train.shape[0])])
    else:
        data_train = np.array(data_train)
        y_train_All = np.array([[0 for x in range(6)] for y in range(len(data_train))])


    if data_test_file == "gendata/bow_test_mat.pkl" or data_test_file == "gendata/bow_test_mat.pkl":
        y_test_All = np.array([[0 for x in range(6)] for y in range(data_test.shape[0])])
    else:
        data_test = np.array(data_test)
        y_test_All = np.array([[0 for x in range(6)] for y in range(len(data_test))])



    # Neutral - positive
    knnNeuP_classifier = KNeighborsClassifier(n_neighbors=knn_n, algorithm=knn_algorithm)

    knnNeuP_classifier.fit(data_train[NeuP_rows], y_train[NeuP_rows])

    y_train_All[:,range(2)] = knnNeuP_classifier.predict_proba(data_train)
    y_test_All[:,range(2)] = knnNeuP_classifier.predict_proba(data_test)
    
    knnNeuP_classifier = []



    # Neutral - negative
    knnNeuNeg_classifier = KNeighborsClassifier(n_neighbors=knn_n, algorithm=knn_algorithm)

    knnNeuNeg_classifier.fit(data_train[NeuNeg_rows], y_train[NeuNeg_rows])

    y_train_All[:,range(2,4,1)] = knnNeuNeg_classifier.predict_proba(data_train)
    y_test_All[:,range(2,4,1)] = knnNeuNeg_classifier.predict_proba(data_test)
    
    knnNeuNeg_classifier = []



    # Positive - negative
    knnPNeg_classifier = KNeighborsClassifier(n_neighbors=knn_n, algorithm=knn_algorithm)

    knnPNeg_classifier.fit(data_train[PNeg_rows], y_train[PNeg_rows])

    y_train_All[:,range(4,6,1)] = knnPNeg_classifier.predict_proba(data_train)
    y_test_All[:,range(4,6,1)] = knnPNeg_classifier.predict_proba(data_test)
    
    knnPNeg_classifier = []




    # Total
    knnT_classifier = KNeighborsClassifier(n_neighbors=knn_n, algorithm=knn_algorithm)

    knnT_classifier.fit(y_train_All, y_train)

    y_test = knnT_classifier.predict(y_test_All)
    
    knnT_classifier = []


    ############# Round-robin classification: probs
 
        
    #
    pickle.dump(y_test, open(y_test_filename, "wb"))
    

################ END FUNCTIONS
    
    
    
    
    


## Round-robin classification

def rr_probs(knn_n=3, knn_algorithm='auto'):
      
    if 1 == knn_n:

        rr_probs_class(bow_mat_filename, bow_test_mat_filename, y_pred_rrp1_bow_file, knn_n, knn_algorithm)            
        rr_probs_class(tfidf_mat_filename, tfidf_test_mat_filename, y_pred_rrp1_tfidf_file, knn_n, knn_algorithm)
        rr_probs_class(emb0_mat_filename, emb0_test_mat_filename, y_pred_rrp1_ebmb0_file, knn_n, knn_algorithm)
        rr_probs_class(emb1_mat_filename, emb1_test_mat_filename, y_pred_rrp1_emb1_file, knn_n, knn_algorithm)
        rr_probs_class(extrafeatures_mat_filename, extrafeatures_test_mat_filename, y_pred_rrp1_extra_file, knn_n, knn_algorithm)
        rr_probs_class(embextra0_mat_filename, embextra0_test_mat_filename, y_pred_rrp1_embextra0_file, knn_n, knn_algorithm)
        rr_probs_class(embextra1_mat_filename, embextra1_test_mat_filename, y_pred_rrp1_embextra1_file, knn_n, knn_algorithm)
               
    
    elif 3 == knn_n:

        rr_probs_class(bow_mat_filename, bow_test_mat_filename, y_pred_rrp3_bow_file, knn_n, knn_algorithm)   
        rr_probs_class(tfidf_mat_filename, tfidf_test_mat_filename, y_pred_rrp3_tfidf_file, knn_n, knn_algorithm)
        rr_probs_class(emb0_mat_filename, emb0_test_mat_filename, y_pred_rrp3_ebmb0_file, knn_n, knn_algorithm)
        rr_probs_class(emb1_mat_filename, emb1_test_mat_filename, y_pred_rrp3_emb1_file, knn_n, knn_algorithm)
        rr_probs_class(extrafeatures_mat_filename, extrafeatures_test_mat_filename, y_pred_rrp3_extra_file, knn_n, knn_algorithm)
        rr_probs_class(embextra0_mat_filename, embextra0_test_mat_filename, y_pred_rrp3_embextra0_file, knn_n, knn_algorithm)
        rr_probs_class(embextra1_mat_filename, embextra1_test_mat_filename, y_pred_rrp3_embextra1_file, knn_n, knn_algorithm)   





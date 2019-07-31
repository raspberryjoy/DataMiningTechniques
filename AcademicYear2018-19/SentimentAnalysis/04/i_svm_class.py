#!/usr/bin/env python
# -*- coding: utf-8 -*-


################ IMPORTS

import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from aux import *




################ FUNCTIONS

def svm_class(data_file, data_test_file, y_test_filename, svm_c=1.0, svm_kernel='rbf', svm_degree=3, svm_max_iter=200):            

    y_train = pickle.load(open(y_train_filename,"rb"))

                                  
    data_train = pickle.load(open(data_file, "rb"))
    data_test = pickle.load(open(data_test_file, "rb")) 

         

    # SVM classifier    

    svm_classifier = SVC(C=svm_c, 
                        kernel=svm_kernel, 
                        degree=svm_degree,  
                        max_iter=svm_max_iter,
                        gamma='auto')                                 


    svm_classifier.fit(data_train, y_train)
    y_test = svm_classifier.predict(data_test)

    data_train = []; data_test = []


    #
    pickle.dump(y_test, open(y_test_filename, "wb"))
    

################ END FUNCTIONS 
    
            
    


## SVM

def svm(svm_c=1.0, svm_kernel='rbf', svm_degree=3, svm_max_iter=500):

    
    if "poly" == svm_kernel:
        
        if 2 == svm_degree:
            
            svm_class(bow_mat_filename, bow_test_mat_filename, y_pred_svmpoly2nd_bow_file, svm_c, svm_kernel, svm_degree, svm_max_iter)    
            svm_class(tfidf_mat_filename, tfidf_test_mat_filename, y_pred_svmpoly2nd_tfidf_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
            svm_class(emb0_mat_filename, emb0_test_mat_filename, y_pred_svmpoly2nd_ebmb0_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
            svm_class(emb1_mat_filename, emb1_test_mat_filename, y_pred_svmpoly2nd_emb1_file, svm_c, svm_kernel, svm_degree, svm_max_iter)     
            svm_class(extrafeatures_mat_filename, extrafeatures_test_mat_filename, y_pred_svmpoly2nd_extra_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
            svm_class(embextra0_mat_filename, embextra0_test_mat_filename, y_pred_svmpoly2nd_embextra0_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
            svm_class(embextra1_mat_filename, embextra1_test_mat_filename, y_pred_svmpoly2nd_embextra1_file, svm_c, svm_kernel, svm_degree, svm_max_iter)     
                        
                
        elif 3 == svm_degree:
                  
            svm_class(bow_mat_filename, bow_test_mat_filename, y_pred_svmpoly3rd_bow_file, svm_c, svm_kernel, svm_degree, svm_max_iter)    
            svm_class(tfidf_mat_filename, tfidf_test_mat_filename, y_pred_svmpoly3rd_tfidf_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
            svm_class(emb0_mat_filename, emb0_test_mat_filename, y_pred_svmpoly3rd_ebmb0_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
            svm_class(emb1_mat_filename, emb1_test_mat_filename, y_pred_svmpoly3rd_emb1_file, svm_c, svm_kernel, svm_degree, svm_max_iter)     
            svm_class(extrafeatures_mat_filename, extrafeatures_test_mat_filename, y_pred_svmpoly3rd_extra_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
            svm_class(embextra0_mat_filename, embextra0_test_mat_filename, y_pred_svmpoly3rd_embextra0_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
            svm_class(embextra1_mat_filename, embextra1_test_mat_filename, y_pred_svmpoly3rd_embextra1_file, svm_c, svm_kernel, svm_degree, svm_max_iter)                                        
        
    
    elif "linear" == svm_kernel:

        svm_class(bow_mat_filename, bow_test_mat_filename, y_pred_svmlin_bow_file, svm_c, svm_kernel, svm_degree, svm_max_iter)    
        svm_class(tfidf_mat_filename, tfidf_test_mat_filename, y_pred_svmlin_tfidf_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(emb0_mat_filename, emb0_test_mat_filename, y_pred_svmlin_ebmb0_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(emb1_mat_filename, emb1_test_mat_filename, y_pred_svmlin_emb1_file, svm_c, svm_kernel, svm_degree, svm_max_iter)     
        svm_class(extrafeatures_mat_filename, extrafeatures_test_mat_filename, y_pred_svmlin_extra_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(embextra0_mat_filename, embextra0_test_mat_filename, y_pred_svmlin_embextra0_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(embextra1_mat_filename, embextra1_test_mat_filename, y_pred_svmlin_embextra1_file, svm_c, svm_kernel, svm_degree, svm_max_iter)   
                   
    
    elif "rbf" == svm_kernel:
         
        svm_class(bow_mat_filename, bow_test_mat_filename, y_pred_svmrbf_bow_file, svm_c, svm_kernel, svm_degree, svm_max_iter)    
        svm_class(tfidf_mat_filename, tfidf_test_mat_filename, y_pred_svmrbf_tfidf_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(emb0_mat_filename, emb0_test_mat_filename, y_pred_svmrbf_ebmb0_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(emb1_mat_filename, emb1_test_mat_filename, y_pred_svmrbf_emb1_file, svm_c, svm_kernel, svm_degree, svm_max_iter)     
        svm_class(extrafeatures_mat_filename, extrafeatures_test_mat_filename, y_pred_svmrbf_extra_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(embextra0_mat_filename, embextra0_test_mat_filename, y_pred_svmrbf_embextra0_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(embextra1_mat_filename, embextra1_test_mat_filename, y_pred_svmrbf_embextra1_file, svm_c, svm_kernel, svm_degree, svm_max_iter)    
                                
    
    elif "sigmoid" == svm_kernel:     
    
        svm_class(bow_mat_filename, bow_test_mat_filename, y_pred_svmsig_bow_file, svm_c, svm_kernel, svm_degree, svm_max_iter)    
        svm_class(tfidf_mat_filename, tfidf_test_mat_filename, y_pred_svmsig_tfidf_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(emb0_mat_filename, emb0_test_mat_filename, y_pred_svmsig_ebmb0_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(emb1_mat_filename, emb1_test_mat_filename, y_pred_svmsig_emb1_file, svm_c, svm_kernel, svm_degree, svm_max_iter)     
        svm_class(extrafeatures_mat_filename, extrafeatures_test_mat_filename, y_pred_svmsig_extra_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(embextra0_mat_filename, embextra0_test_mat_filename, y_pred_svmsig_embextra0_file, svm_c, svm_kernel, svm_degree, svm_max_iter) 
        svm_class(embextra1_mat_filename, embextra1_test_mat_filename, y_pred_svmsig_embextra1_file, svm_c, svm_kernel, svm_degree, svm_max_iter)                          





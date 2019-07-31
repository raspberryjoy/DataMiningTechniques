#!/usr/bin/env python
# -*- coding: utf-8 -*-



################ IMPORTS

import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA

from aux import *





################ FUNCTIONS

def pca_array(data_mat_filename, pca_data_mat_filename, pca_dims=3, pca_whiten=False, svd_solver='auto'):

    data_mat = n.array(pickle.load(open(data_mat_filename, "rb")))

    pca = PCA(n_components=pca_dims, 
                whiten=pca_whiten,
                svd_solver=svd_solver)
                                        
    pca_res = pca.fit_transform(data_mat.toarray()) 
    
    # Save the data
    pickle.dump(pca_res, open(pca_data_mat_filename, "wb"))  
    
    
    
def pca_mat(data_mat_filename, pca_data_mat_filename, pca_dims=3, pca_whiten=False, svd_solver='auto'):

    data_mat = np.array(pickle.load(open(data_mat_filename, "rb")))

    pca = PCA(n_components=pca_dims, 
                whiten=pca_whiten,
                svd_solver=svd_solver)
                                        
    pca_res = pca.fit_transform(data_mat) 
    
    # Save the data
    pickle.dump(pca_res, open(pca_data_mat_filename, "wb"))       



def sparsepca_array(data_mat_filename, pca_data_mat_filename, pca_dims=3):

    data_mat = pickle.load(open(data_mat_filename, "rb"))

    pca = SparsePCA(n_components=pca_dims,
                    normalize_components=True)
                                        
    pca_res = pca.fit_transform(data_mat.toarray()) 
    
    # Save the data
    pickle.dump(pca_res, open(pca_data_mat_filename, "wb"))  
    
    
    
def sparsepca_mat(data_mat_filename, pca_data_mat_filename, pca_dims=3):

    data_mat = pickle.load(open(data_mat_filename, "rb"))

    pca = SparsePCA(n_components=pca_dims,
                    normalize_components=True)
                                        
    pca_res = pca.fit_transform(data_mat) 
    
    # Save the data
    pickle.dump(pca_res, open(pca_data_mat_filename, "wb")) 



################ END FUNCTIONS
    

    
    
    

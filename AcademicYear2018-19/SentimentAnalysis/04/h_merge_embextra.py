#!/usr/bin/env python
# -*- coding: utf-8 -*-



################ IMPORTS

import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle

from aux import *
#from pca_funs import pca_mat



################ FUNCTIONS

def embextra(emb_mat_filename, extrafeatures_mat_filename, embextra_mat_filename, pca_embextra_mat_filename, pca_dims=3, pca_whiten=True):

    #
    emb_mat = pickle.load(open(emb_mat_filename, "rb"))
    extrafeatures_mat = pickle.load(open(extrafeatures_mat_filename, "rb"))

    # Word embedding + extra features
    embextra_mat = np.array([[0 for x in range(len(emb_mat[0]) + len(extrafeatures_mat[0]))] for y in range(len(emb_mat))]) 
    embextra_mat[:,range(len(emb_mat[0]))] = emb_mat
    embextra_mat[:,range(len(emb_mat[0]), len(embextra_mat[0]), 1)] = extrafeatures_mat
    
    
    # Save the data
    pickle.dump(embextra_mat, open(embextra_mat_filename, "wb"))


    # PCA
    #pca_mat(embextra_mat_filename, pca_embextra_mat_filename, pca_dims, pca_whiten)
    

################ END FUNCTIONS






## Extra features
def merge_embextra():

    embextra(emb0_mat_filename, extrafeatures_mat_filename, embextra0_mat_filename, pca_embextra0_mat_filename)
    embextra(emb0_test_mat_filename, extrafeatures_test_mat_filename, embextra0_test_mat_filename, pca_embextra0_test_mat_filename)
    
    embextra(emb1_mat_filename, extrafeatures_mat_filename, embextra1_mat_filename, pca_embextra1_mat_filename)
    embextra(emb1_test_mat_filename, extrafeatures_test_mat_filename, embextra1_test_mat_filename, pca_embextra1_test_mat_filename)
        
    
   


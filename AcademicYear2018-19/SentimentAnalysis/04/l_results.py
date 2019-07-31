#!/usr/bin/env python
# -*- coding: utf-8 -*-



################ IMPORTS

from decimal import Decimal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import six
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import cPickle as pickle

from aux import *




################ FUNCTIONS

def conf_perc(y_test_corr_filename, y_pred_file):

    y_true = pickle.load(open(y_test_corr_filename,"rb"))
    y_pred = pickle.load(open(y_pred_file,"rb"))
    
    
    # Compute confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)
    
    return([round(Decimal(1.0 * conf_mat[0,0] / np.sum(conf_mat[0,:])), 2), 
            round(Decimal(1.0 * conf_mat[1,1] / np.sum(conf_mat[1,:])), 2),
            round(Decimal(1.0 * conf_mat[2,2] / np.sum(conf_mat[2,:])), 2)])
        
        
    
def get_f1_score(y_test_corr_filename, y_pred_file):

    y_true = np.array(pickle.load(open(y_test_corr_filename,"rb")))
    y_pred = np.array(pickle.load(open(y_pred_file,"rb")))   

    return round(Decimal(f1_score(y_true, y_pred, average='weighted')), 2)


# source(edited): https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
def plot_heatmap(data_arr, row_names, col_names, title, save_filename):

    col_names = col_names
    row_names = row_names

    fig, ax = plt.subplots()
    im = ax.imshow(data_arr)


    # We want to show all ticks...
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_yticks(np.arange(len(row_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(col_names)
    ax.set_yticklabels(row_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_names)):
        for j in range(len(col_names)):
            text = ax.text(j, i, data_arr[i,j], ha="center", va="center", color="w")

    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_filename)

################ END FUNCTIONS

           





def results():

    # Results' summary

    results_df = pd.DataFrame()

    results_df['SVM_linear'] = [get_f1_score(y_test_corr_filename, y_pred_svmlin_bow_file), get_f1_score(y_test_corr_filename, y_pred_svmlin_tfidf_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmlin_ebmb0_file), get_f1_score(y_test_corr_filename, y_pred_svmlin_emb1_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmlin_extra_file), get_f1_score(y_test_corr_filename, y_pred_svmlin_embextra0_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmlin_embextra1_file)]
    results_df['SVM_poly_2nd'] = [get_f1_score(y_test_corr_filename, y_pred_svmpoly2nd_bow_file), get_f1_score(y_test_corr_filename, y_pred_svmpoly2nd_tfidf_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmpoly2nd_ebmb0_file), get_f1_score(y_test_corr_filename, y_pred_svmpoly2nd_emb1_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmpoly2nd_extra_file), get_f1_score(y_test_corr_filename, y_pred_svmpoly2nd_embextra0_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmpoly2nd_embextra1_file)]
    results_df['SVM_poly_3rd'] = [get_f1_score(y_test_corr_filename, y_pred_svmpoly3rd_bow_file), get_f1_score(y_test_corr_filename, y_pred_svmpoly3rd_tfidf_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmpoly3rd_ebmb0_file), get_f1_score(y_test_corr_filename, y_pred_svmpoly3rd_emb1_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmpoly3rd_extra_file), get_f1_score(y_test_corr_filename, y_pred_svmpoly3rd_embextra0_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmpoly3rd_embextra1_file)]
    results_df['SVM_rbf'] = [get_f1_score(y_test_corr_filename, y_pred_svmrbf_bow_file), get_f1_score(y_test_corr_filename, y_pred_svmrbf_tfidf_file), 
                                get_f1_score(y_test_corr_filename, y_pred_svmrbf_ebmb0_file), get_f1_score(y_test_corr_filename, y_pred_svmrbf_emb1_file), 
                                get_f1_score(y_test_corr_filename, y_pred_svmrbf_extra_file), get_f1_score(y_test_corr_filename, y_pred_svmrbf_embextra0_file), 
                                get_f1_score(y_test_corr_filename, y_pred_svmrbf_embextra1_file)]
    results_df['SVM_sigmoid'] = [get_f1_score(y_test_corr_filename, y_pred_svmsig_bow_file), get_f1_score(y_test_corr_filename, y_pred_svmsig_tfidf_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmsig_ebmb0_file), get_f1_score(y_test_corr_filename, y_pred_svmsig_emb1_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmsig_extra_file), get_f1_score(y_test_corr_filename, y_pred_svmsig_embextra0_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_svmsig_embextra1_file)]
    results_df['kNN_1'] = [get_f1_score(y_test_corr_filename, y_pred_knn1_bow_file), get_f1_score(y_test_corr_filename, y_pred_knn1_tfidf_file), 
                                get_f1_score(y_test_corr_filename, y_pred_knn1_ebmb0_file), get_f1_score(y_test_corr_filename, y_pred_knn1_emb1_file), 
                                get_f1_score(y_test_corr_filename, y_pred_knn1_extra_file), get_f1_score(y_test_corr_filename, y_pred_knn1_embextra0_file), 
                                get_f1_score(y_test_corr_filename, y_pred_knn1_embextra1_file)]
    results_df['kNN_3'] = [get_f1_score(y_test_corr_filename, y_pred_knn3_bow_file), get_f1_score(y_test_corr_filename, y_pred_knn3_tfidf_file), 
                                get_f1_score(y_test_corr_filename, y_pred_knn3_ebmb0_file), get_f1_score(y_test_corr_filename, y_pred_knn3_emb1_file), 
                                get_f1_score(y_test_corr_filename, y_pred_knn3_extra_file), get_f1_score(y_test_corr_filename, y_pred_knn3_embextra0_file), 
                                get_f1_score(y_test_corr_filename, y_pred_knn3_embextra1_file)]
    results_df['RR_1_class'] = [get_f1_score(y_test_corr_filename, y_pred_rr1_bow_file), get_f1_score(y_test_corr_filename, y_pred_rr1_tfidf_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rr1_ebmb0_file), get_f1_score(y_test_corr_filename, y_pred_rr1_emb1_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rr1_extra_file), get_f1_score(y_test_corr_filename, y_pred_rr1_embextra0_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rr1_embextra1_file)]
    results_df['RR_3_class'] = [get_f1_score(y_test_corr_filename, y_pred_rr3_bow_file), get_f1_score(y_test_corr_filename, y_pred_rr3_tfidf_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rr3_ebmb0_file), get_f1_score(y_test_corr_filename, y_pred_rr3_emb1_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rr3_extra_file), get_f1_score(y_test_corr_filename, y_pred_rr3_embextra0_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rr3_embextra1_file)]
    results_df['RR_1_probs'] = [get_f1_score(y_test_corr_filename, y_pred_rrp1_bow_file), get_f1_score(y_test_corr_filename, y_pred_rrp1_tfidf_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rrp1_ebmb0_file), get_f1_score(y_test_corr_filename, y_pred_rrp1_emb1_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rrp1_extra_file), get_f1_score(y_test_corr_filename, y_pred_rrp1_embextra0_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rrp1_embextra1_file)]
    results_df['RR_3_probs'] = [get_f1_score(y_test_corr_filename, y_pred_rrp3_bow_file), get_f1_score(y_test_corr_filename, y_pred_rrp3_tfidf_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rrp3_ebmb0_file), get_f1_score(y_test_corr_filename, y_pred_rrp3_emb1_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rrp3_extra_file), get_f1_score(y_test_corr_filename, y_pred_rrp3_embextra0_file), 
                                    get_f1_score(y_test_corr_filename, y_pred_rrp3_embextra1_file)]



    ################## F-score
    
    fscore_arr = np.transpose(np.array([results_df['SVM_linear'],
                            results_df['SVM_poly_2nd'],
                            results_df['SVM_poly_3rd'],
                            results_df['SVM_rbf'],
                            results_df['SVM_sigmoid'],
                            results_df['kNN_1'],
                            results_df['kNN_3'],
                            results_df['RR_1_class'],
                            results_df['RR_3_class'],
                            results_df['RR_1_probs'],
                            results_df['RR_1_probs']]))
  

    plot_heatmap(fscore_arr, ['BOW', 'TF-IDF', 'WE-CBOW', 'WE-SG', 'Extra feat.', 'WE-CBOW & extra feat.', 'WE-SG & extra feat.'], 
                results_df.columns, "F-score summary", img_output_dir + "/fscore.png")


    ################## Ratios of correct classification (best case ~ F1-score)   

    y_pred_svmlin_best = y_pred_svmlin[np.where(results_df['SVM_linear'] == max(results_df['SVM_linear']))[0][0]]
    y_pred_svmpoly2_best = y_pred_svmpoly2[np.where(results_df['SVM_poly_2nd'] == max(results_df['SVM_poly_2nd']))[0][0]]
    y_pred_svmpoly3_best = y_pred_svmpoly3[np.where(results_df['SVM_poly_3rd'] == max(results_df['SVM_poly_3rd']))[0][0]]
    y_pred_svmrbf_best = y_pred_svmrbf[np.where(results_df['SVM_rbf'] == max(results_df['SVM_rbf']))[0][0]]
    y_pred_svmsig_best = y_pred_svmsig[np.where(results_df['SVM_sigmoid'] == max(results_df['SVM_sigmoid']))[0][0]]
    y_pred_knn1_best = y_pred_knn1[np.where(results_df['kNN_1'] == max(results_df['kNN_1']))[0][0]]
    y_pred_knn3_best = y_pred_knn3[np.where(results_df['kNN_3'] == max(results_df['kNN_3']))[0][0]]
    y_pred_rr1_best = y_pred_rr1[np.where(results_df['RR_1_class'] == max(results_df['RR_1_class']))[0][0]]
    y_pred_rr3_best = y_pred_rr3[np.where(results_df['RR_3_class'] == max(results_df['RR_3_class']))[0][0]]
    y_pred_rrp1_best = y_pred_rrp1[np.where(results_df['RR_1_probs'] == max(results_df['RR_1_probs']))[0][0]]
    y_pred_rrp3_best = y_pred_rrp3[np.where(results_df['RR_3_probs'] == max(results_df['RR_3_probs']))[0][0]]


    classratio_arr = np.transpose(np.array([ conf_perc(y_test_corr_filename, y_pred_svmlin_best),
                         conf_perc(y_test_corr_filename, y_pred_svmpoly2_best),
                         conf_perc(y_test_corr_filename, y_pred_svmpoly3_best),
                         conf_perc(y_test_corr_filename, y_pred_svmrbf_best),
                         conf_perc(y_test_corr_filename, y_pred_svmsig_best),
                         conf_perc(y_test_corr_filename, y_pred_knn1_best),
                         conf_perc(y_test_corr_filename, y_pred_knn3_best),
                         conf_perc(y_test_corr_filename, y_pred_rr1_best),
                         conf_perc(y_test_corr_filename, y_pred_rr3_best),
                         conf_perc(y_test_corr_filename, y_pred_rrp1_best),
                         conf_perc(y_test_corr_filename, y_pred_rrp3_best)]))



    plot_heatmap(classratio_arr, ['Negative', 'Neutral', 'Positive'], results_df.columns, 
                "Ratios of correct classification (best case ~ F1-score)", img_output_dir + "/classratio.png")





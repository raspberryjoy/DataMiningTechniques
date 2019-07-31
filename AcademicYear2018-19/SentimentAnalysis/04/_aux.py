#!/usr/bin/env python
# -*- coding: utf-8 -*-



################ IMPORTS

from os import mkdir, path
import cPickle as pickle
import pandas as pd
import numpy as np
import re
from collections import Counter



# Filenames
bow_mat_filename = "gendata/bow_mat.pkl"
bow_test_mat_filename = "gendata/bow_test_mat.pkl"
bow_y_test_filename = "gendata/knn_bow_y_test.pkl"
bow_y_test_filename = "gendata/rr_bow_y_test.pkl"
bow_y_test_filename = "gendata/rr_probs_bow_y_test.pkl"
bow_y_test_filename = "gendata/svm_bow_y_test.pkl"
emb0_mat_filename = "gendata/emb0_mat.pkl"
emb0_test_mat_filename = "gendata/emb0_test_mat.pkl"
emb0_y_test_filename = "gendata/knn_emb0_y_test.pkl"
emb0_y_test_filename = "gendata/rr_emb0_y_test.pkl"
emb0_y_test_filename = "gendata/rr_probs_emb0_y_test.pkl"
emb0_y_test_filename = "gendata/svm_emb0_y_test.pkl"
emb1_mat_filename = "gendata/emb1_mat.pkl"
emb1_test_mat_filename = "gendata/emb1_test_mat.pkl"
emb1_y_test_filename = "gendata/knn_emb1_y_test.pkl"
emb1_y_test_filename = "gendata/rr_emb1_y_test.pkl"
emb1_y_test_filename = "gendata/rr_probs_emb1_y_test.pkl"
emb1_y_test_filename = "gendata/svm_emb1_y_test.pkl"
embextra0_mat_filename = "gendata/embextra0_mat.pkl"
embextra0_test_mat_filename = "gendata/embextra0_test_mat.pkl"
embextra0_y_test_filename = "gendata/knn_embextra0_y_test.pkl"
embextra0_y_test_filename = "gendata/rr_embextra0_y_test.pkl"
embextra0_y_test_filename = "gendata/svm_embextra0_y_test.pkl"
embextra1_mat_filename = "gendata/embextra1_mat.pkl"
embextra1_test_mat_filename = "gendata/embextra1_test_mat.pkl"
embextra1_y_test_filename = "gendata/knn_embextra1_y_test.pkl"
embextra1_y_test_filename = "gendata/rr_embextra1_y_test.pkl"
embextra1_y_test_filename = "gendata/rr_probs_embextra1_y_test.pkl"
embextra1_y_test_filename = "gendata/svm_embextra1_y_test.pkl"
extrafeatures_mat_filename = "gendata/extrafeatures_mat.pkl"
extrafeatures_test_mat_filename = "gendata/extrafeatures_test_mat.pkl"
extrafeatures_y_test_filename = "gendata/knn_extrafeatures_y_test.pkl"
extrafeatures_y_test_filename = "gendata/rr_extrafeatures_y_test.pkl"
extrafeatures_y_test_filename = "gendata/rr_probs_extrafeatures_y_test.pkl"
extrafeatures_y_test_filename = "gendata/svm_extrafeatures_y_test.pkl"
file_output_dir = "gendata"
img_output_dir = "gendata/img"
pca_bow_mat_filename = "gendata/pca_bow_mat.pkl"
pca_bow_test_mat_filename = "gendata/pca_bow_test_mat.pkl"
pca_emb0_mat_filename = "gendata/pca_emb0_mat.pkl"
pca_emb0_test_mat_filename = "gendata/pca_emb0_test_mat.pkl"
pca_emb1_mat_filename = "gendata/pca_emb1_mat.pkl"
pca_emb1_test_mat_filename = "gendata/pca_emb1_test_mat.pkl"
pca_embextra0_mat_filename = "gendata/pca_embextra0_mat.pkl"
pca_embextra0_test_mat_filename = "gendata/pca_embextra0_test_mat.pkl"
pca_embextra1_mat_filename = "gendata/pca_embextra1_mat.pkl"
pca_embextra1_test_mat_filename = "gendata/pca_embextra1_test_mat.pkl"
pca_extrafeatures_mat_filename = "gendata/pca_extrafeatures_mat.pkl"
pca_extrafeatures_test_mat_filename = "gendata/pca_extrafeatures_test_mat.pkl"
pca_tfidf_mat_filename = "gendata/pca_tfidf_mat.pkl"
pca_tfidf_test_mat_filename = "gendata/pca_tfidf_test_mat.pkl"
test_data_class_filename = "../twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt"
test_data_filename = "../twitter_data/test2017.tsv"
test_df_prep_filename = "gendata/test_df_prep.pkl"
test_init_filename = "../twitter_data/test2017.tsv"
tfidf_mat_filename = "gendata/tfidf_mat.pkl"
tfidf_test_mat_filename = "gendata/tfidf_test_mat.pkl"
tfidf_y_test_filename = "gendata/knn_tfidf_y_test.pkl"
tfidf_y_test_filename = "gendata/rr_probs_tfidf_y_test.pkl"
tfidf_y_test_filename = "gendata/rr_tfidf_y_test.pkl"
tfidf_y_test_filename = "gendata/svm_tfidf_y_test.pkl"
toklemm_tweets_filename = "gendata/toklemm_words.pkl"
toklemm_tweets_test_filename = "gendata/toklemm_test_words.pkl"
tokstem_tweets_filename = "gendata/tokstem_words.pkl"
tokstem_tweets_test_filename = "gendata/tokstem_test_words.pkl"
train_data_filename = "../twitter_data/train2017.tsv"
train_df_prep_filename = "gendata/train_df_prep.pkl"
train_init_filename = "../twitter_data/train2017.tsv"
y_pred_knn1_bow_file = "gendata/y_pred_knn1_bow_file"
y_pred_knn1_ebmb0_file = "gendata/y_pred_knn1_ebmb0_file"
y_pred_knn1_emb1_file = "gendata/y_pred_knn1_emb1_file"
y_pred_knn1_embextra0_file = "gendata/y_pred_knn1_embextra0_file"
y_pred_knn1_embextra1_file = "gendata/y_pred_knn1_embextra1_file"
y_pred_knn1_extra_file = "gendata/y_pred_knn1_extra_file"
y_pred_knn1_tfidf_file = "gendata/y_pred_knn1_tfidf_file"
y_pred_knn3_bow_file = "gendata/y_pred_knn3_bow_file"
y_pred_knn3_ebmb0_file = "gendata/y_pred_knn3_ebmb0_file"
y_pred_knn3_emb1_file = "gendata/y_pred_knn3_emb1_file"
y_pred_knn3_embextra0_file = "gendata/y_pred_knn3_embextra0_file"
y_pred_knn3_embextra1_file = "gendata/y_pred_knn3_embextra1_file"
y_pred_knn3_extra_file = "gendata/y_pred_knn3_extra_file"
y_pred_knn3_tfidf_file = "gendata/y_pred_knn3_tfidf_file"
y_pred_rr1_bow_file = "gendata/y_pred_rr1_bow_file"
y_pred_rr1_ebmb0_file = "gendata/y_pred_rr1_ebmb0_file"
y_pred_rr1_emb1_file = "gendata/y_pred_rr1_emb1_file"
y_pred_rr1_embextra0_file = "gendata/y_pred_rr1_embextra0_file"
y_pred_rr1_embextra1_file = "gendata/y_pred_rr1_embextra1_file"
y_pred_rr1_extra_file = "gendata/y_pred_rr1_extra_file"
y_pred_rr1_tfidf_file = "gendata/y_pred_rr1_tfidf_file"
y_pred_rr3_bow_file = "gendata/y_pred_rr3_bow_file"
y_pred_rr3_ebmb0_file = "gendata/y_pred_rr3_ebmb0_file"
y_pred_rr3_emb1_file = "gendata/y_pred_rr3_emb1_file"
y_pred_rr3_embextra0_file = "gendata/y_pred_rr3_embextra0_file"
y_pred_rr3_embextra1_file = "gendata/y_pred_rr3_embextra1_file"
y_pred_rr3_extra_file = "gendata/y_pred_rr3_extra_file"
y_pred_rr3_tfidf_file = "gendata/y_pred_rr3_tfidf_file"
y_pred_rrp1_bow_file = "gendata/y_pred_rrp1_bow_file"
y_pred_rrp1_ebmb0_file = "gendata/y_pred_rrp1_ebmb0_file"
y_pred_rrp1_emb1_file = "gendata/y_pred_rrp1_emb1_file"
y_pred_rrp1_embextra0_file = "gendata/y_pred_rrp1_embextra0_file"
y_pred_rrp1_embextra1_file = "gendata/y_pred_rrp1_embextra1_file"
y_pred_rrp1_extra_file = "gendata/y_pred_rrp1_extra_file"
y_pred_rrp1_tfidf_file = "gendata/y_pred_rrp1_tfidf_file"
y_pred_rrp3_bow_file = "gendata/y_pred_rrp3_bow_file"
y_pred_rrp3_ebmb0_file = "gendata/y_pred_rrp3_ebmb0_file"
y_pred_rrp3_emb1_file = "gendata/y_pred_rrp3_emb1_file"
y_pred_rrp3_embextra0_file = "gendata/y_pred_rrp3_embextra0_file"
y_pred_rrp3_embextra1_file = "gendata/y_pred_rrp3_embextra1_file"
y_pred_rrp3_extra_file = "gendata/y_pred_rrp3_extra_file"
y_pred_rrp3_tfidf_file = "gendata/y_pred_rrp3_tfidf_file"
y_pred_svmlin_bow_file = "gendata/y_pred_svmlin_bow_file"
y_pred_svmlin_ebmb0_file = "gendata/y_pred_svmlin_ebmb0_file"
y_pred_svmlin_emb1_file = "gendata/y_pred_svmlin_emb1_file"
y_pred_svmlin_embextra0_file = "gendata/y_pred_svmlin_embextra0_file"
y_pred_svmlin_embextra1_file = "gendata/y_pred_svmlin_embextra1_file"
y_pred_svmlin_extra_file = "gendata/y_pred_svmlin_extra_file"
y_pred_svmlin_tfidf_file = "gendata/y_pred_svmlin_tfidf_file"
y_pred_svmpoly2nd_bow_file = "gendata/y_pred_svmpoly2nd_bow_file"
y_pred_svmpoly2nd_ebmb0_file = "gendata/y_pred_svmpoly2nd_ebmb0_file"
y_pred_svmpoly2nd_emb1_file = "gendata/y_pred_svmpoly2nd_emb1_file"
y_pred_svmpoly2nd_embextra0_file = "gendata/y_pred_svmpoly2nd_embextra0_file"
y_pred_svmpoly2nd_embextra1_file = "gendata/y_pred_svmpoly2nd_embextra1_file"
y_pred_svmpoly2nd_extra_file = "gendata/y_pred_svmpoly2nd_extra_file"
y_pred_svmpoly2nd_tfidf_file = "gendata/y_pred_svmpoly2nd_tfidf_file"
y_pred_svmpoly3rd_bow_file = "gendata/y_pred_svmpoly3rd_bow_file"
y_pred_svmpoly3rd_ebmb0_file = "gendata/y_pred_svmpoly3rd_ebmb0_file"
y_pred_svmpoly3rd_emb1_file = "gendata/y_pred_svmpoly3rd_emb1_file"
y_pred_svmpoly3rd_embextra0_file = "gendata/y_pred_svmpoly3rd_embextra0_file"
y_pred_svmpoly3rd_embextra1_file = "gendata/y_pred_svmpoly3rd_embextra1_file"
y_pred_svmpoly3rd_extra_file = "gendata/y_pred_svmpoly3rd_extra_file"
y_pred_svmpoly3rd_tfidf_file = "gendata/y_pred_svmpoly3rd_tfidf_file"
y_pred_svmrbf_bow_file = "gendata/y_pred_svmrbf_bow_file"
y_pred_svmrbf_ebmb0_file = "gendata/y_pred_svmrbf_ebmb0_file"
y_pred_svmrbf_emb1_file = "gendata/y_pred_svmrbf_emb1_file"
y_pred_svmrbf_embextra0_file = "gendata/y_pred_svmrbf_embextra0_file"
y_pred_svmrbf_embextra1_file = "gendata/y_pred_svmrbf_embextra1_file"
y_pred_svmrbf_extra_file = "gendata/y_pred_svmrbf_extra_file"
y_pred_svmrbf_tfidf_file = "gendata/y_pred_svmrbf_tfidf_file"
y_pred_svmsig_bow_file = "gendata/y_pred_svmsig_bow_file"
y_pred_svmsig_ebmb0_file = "gendata/y_pred_svmsig_ebmb0_file"
y_pred_svmsig_emb1_file = "gendata/y_pred_svmsig_emb1_file"
y_pred_svmsig_embextra0_file = "gendata/y_pred_svmsig_embextra0_file"
y_pred_svmsig_embextra1_file = "gendata/y_pred_svmsig_embextra1_file"
y_pred_svmsig_extra_file = "gendata/y_pred_svmsig_extra_file"
y_pred_svmsig_tfidf_file = "gendata/y_pred_svmsig_tfidf_file"
y_true_file = "gendata/y_true_file"
y_train_filename = "gendata/y_train.pkl"
y_test_corr_filename = "gendata/y_test_corr.pkl"
x_train_neg_filename = "gendata/x_train_neg.pkl"
x_train_neu_filename = "gendata/x_train_neu.pkl"
x_train_pos_filename = "gendata/x_train_pos.pkl"
tweets_hashtags_filename = "gendata/tweets_hashtags.pkl"
tweets_hashtags_test_filename = "gendata/tweets_hashtags_test.pkl"
top20_hashtags_filename = "gendata/top20_hashtags.pkl"
top20_hashtags_pos_filename = "gendata/top20_hashtags_pos.pkl"
top20_hashtags_neg_filename = "gendata/top20_hashtags_neg.pkl"
top20_hashtags_neu_filename = "gendata/top20_hashtags_neu.pkl"




y_pred_svmlin = [y_pred_svmlin_bow_file, y_pred_svmlin_tfidf_file, y_pred_svmlin_ebmb0_file, y_pred_svmlin_emb1_file, 
                y_pred_svmlin_extra_file, y_pred_svmlin_embextra0_file, y_pred_svmlin_embextra1_file]
y_pred_svmpoly2 = [y_pred_svmpoly2nd_bow_file, y_pred_svmpoly2nd_tfidf_file, y_pred_svmpoly2nd_ebmb0_file, y_pred_svmpoly2nd_emb1_file, 
                y_pred_svmpoly2nd_extra_file, y_pred_svmpoly2nd_embextra0_file, y_pred_svmpoly2nd_embextra1_file]
y_pred_svmpoly3 = [y_pred_svmpoly3rd_bow_file, y_pred_svmpoly3rd_tfidf_file, y_pred_svmpoly3rd_ebmb0_file, y_pred_svmpoly3rd_emb1_file, 
                y_pred_svmpoly3rd_extra_file, y_pred_svmpoly3rd_embextra0_file, y_pred_svmpoly3rd_embextra1_file]
y_pred_svmrbf = [y_pred_svmrbf_bow_file, y_pred_svmrbf_tfidf_file, y_pred_svmrbf_ebmb0_file, y_pred_svmrbf_emb1_file, 
                y_pred_svmrbf_extra_file, y_pred_svmrbf_embextra0_file, y_pred_svmrbf_embextra1_file]
y_pred_svmsig = [y_pred_svmsig_bow_file, y_pred_svmsig_tfidf_file, y_pred_svmsig_ebmb0_file, y_pred_svmsig_emb1_file, 
                y_pred_svmsig_extra_file, y_pred_svmsig_embextra0_file, y_pred_svmsig_embextra1_file]
y_pred_knn1 = [y_pred_knn1_bow_file, y_pred_knn1_tfidf_file, y_pred_knn1_ebmb0_file, y_pred_knn1_emb1_file, 
                y_pred_knn1_extra_file, y_pred_knn1_embextra0_file, y_pred_knn1_embextra1_file]
y_pred_knn3 = [y_pred_knn3_bow_file, y_pred_knn3_tfidf_file, y_pred_knn3_ebmb0_file, y_pred_knn3_emb1_file, 
                y_pred_knn3_extra_file, y_pred_knn3_embextra0_file, y_pred_knn3_embextra1_file]
y_pred_rr1 = [y_pred_rr1_bow_file, y_pred_rr1_tfidf_file, y_pred_rr1_ebmb0_file, y_pred_rr1_emb1_file, 
                y_pred_rr1_extra_file, y_pred_rr1_embextra0_file, y_pred_rr1_embextra1_file]
y_pred_rr3 = [y_pred_rr3_bow_file, y_pred_rr3_tfidf_file, y_pred_rr3_ebmb0_file, y_pred_rr3_emb1_file, 
                y_pred_rr3_extra_file, y_pred_rr3_embextra0_file, y_pred_rr3_embextra1_file]
y_pred_rrp1 = [y_pred_rrp1_bow_file, y_pred_rrp1_tfidf_file, y_pred_rrp1_ebmb0_file, y_pred_rrp1_emb1_file, 
                y_pred_rrp1_extra_file, y_pred_rrp1_embextra0_file, y_pred_rrp1_embextra1_file]
y_pred_rrp3 = [y_pred_rrp3_bow_file, y_pred_rrp3_tfidf_file, y_pred_rrp3_ebmb0_file, y_pred_rrp3_emb1_file, 
                y_pred_rrp3_extra_file, y_pred_rrp3_embextra0_file, y_pred_rrp3_embextra1_file]





# Dictionaries
lexica = ["../lexica/affin/affin.txt",
            "../lexica/generic/generic.txt"] # not unique
#            "../lexica/emotweet/valence_tweet.txt",          
#            "../lexica/nrc/val.txt",
#            "../lexica/nrctag/val.txt"]



rownames = ('tweet_len','num_emoticons','num_excl','num_urls','num_hashtags','num_mentions',
            'num_words','num_top_poshash','num_top_neghash','num_top_neuhash', 'num_vowels', 'end')
idx = dict(zip(rownames, range(len(lexica) * 3, len(rownames) + len(lexica) * 3)))


scale_suff = "_scaled"
scaleOut_suff = "_scaledOut"




#######

negative_val = -1
neutral_val = 0
positive_val = 1





def aux():


    ############# Create the files' output directory if it doesn't exist
    if not path.exists(file_output_dir):
        mkdir(file_output_dir)

    if not path.exists(img_output_dir):
        mkdir(img_output_dir)


    ############# Classification (numeric) of the training data
    ############# Indices of the training data per class

    
    # Class per tweet of the training data
    y_train = []
    x_train_neg = []; x_train_neu = []; x_train_pos = []

    # Training data
    train_df = pd.read_csv(train_data_filename, 
                            sep="\t", index_col=False,
                            names=['id1', 'id2', 'class', 'tweet'])    
                

    for i in range(0, len(train_df.index), 1):

        if "negative" == train_df.loc[i,'class']:
            y_train.append(negative_val)
            x_train_neg.append(i)
        elif "neutral" == train_df.loc[i,'class']:
            y_train.append(neutral_val)
            x_train_neu.append(i)
        else:
            y_train.append(positive_val)
            x_train_pos.append(i)


    # Save the data
    pickle.dump(y_train, open(y_train_filename, "wb"))
    pickle.dump(x_train_neg, open(x_train_neg_filename, "wb"))
    pickle.dump(x_train_neu, open(x_train_neu_filename, "wb"))
    pickle.dump(x_train_pos, open(x_train_pos_filename, "wb"))    

    train_df = []; x_train_neg = []; x_train_neu = []; x_train_pos = []; x_train = [] 
    
    
    ############# Correct classification (numeric) of the test data    
    
    # Correct class per tweet of the test data
    y_test_corr = []

    # Test data
    test_df = pd.read_csv(test_data_filename, 
                            sep="\t", index_col=False,
                            names=['id1', 'id2', 'class', 'tweet']) 
                    
    # Correct class per tweet of the test data
    corr_test_y = pd.read_csv(test_data_class_filename, 
                    sep="\t", index_col=False, encoding='utf-8',
                    names=['id1', 'class'])
                                    
            
    for i in range(0, len(test_df.index), 1):

        # Match the ids
        y = np.where(corr_test_y['id1'] == test_df.loc[i,'id1'])[0][0]

        if "negative" == corr_test_y.loc[y,'class']:
            y_test_corr.append(negative_val)
        elif "neutral" == corr_test_y.loc[y,'class']:
            y_test_corr.append(neutral_val)
        else:
            y_test_corr.append(positive_val)                

    
    # Save the data
    pickle.dump(y_test_corr, open(y_test_corr_filename, "wb"))

    test_df = []; corr_test_y = []; y_test_corr = []    
    
    
    ############# Top 15 hashtags (total, per class)
    
    data_df = pd.read_csv(train_data_filename, 
                        sep="\t", index_col=False, encoding='utf-8',
                        names=['id1', 'id2', 'class', 'tweet'])
    
    pattern = re.compile("^#.+$")                    
    tweets_hashtags = np.array(data_df.tweet.apply(lambda x: ' '.join(set([re.sub(r"\\u[a-zA-Z0-9]{4}", "", item) for item in x.split() if pattern.match(item)]))))
    
    pickle.dump(tweets_hashtags, open(tweets_hashtags_filename, "wb"))    
    
    
    
    # top hashtags per class
    neg = pickle.load(open(x_train_neg_filename, "rb"))
    neu = pickle.load(open(x_train_neu_filename, "rb"))
    pos = pickle.load(open(x_train_pos_filename, "rb"))

    # omit the first element -> ''
    counter = Counter(tweets_hashtags)    
    pickle.dump(np.array(counter.most_common(21))[1:20,0], open(top20_hashtags_filename, "wb"))

    counter = Counter(tweets_hashtags[pos])    
    pickle.dump(np.array(counter.most_common(21))[1:20,0], open(top20_hashtags_pos_filename, "wb"))

    counter = Counter(tweets_hashtags[neg])    
    pickle.dump(np.array(counter.most_common(21))[1:20,0], open(top20_hashtags_neg_filename, "wb"))

    counter = Counter(tweets_hashtags[neu])    
    pickle.dump(np.array(counter.most_common(21))[1:20,0], open(top20_hashtags_neu_filename, "wb"))        
    
    
    data_df = []; tweets_hashtags = []; counter = []
    
    
    
    
    data_df = pd.read_csv(test_data_filename, 
                        sep="\t", index_col=False, encoding='utf-8',
                        names=['id1', 'id2', 'class', 'tweet'])
    
    pattern = re.compile("^#.+$")                    
    tweets_hashtags = np.array(data_df.tweet.apply(lambda x: ' '.join(set([re.sub(r"\\u[a-zA-Z0-9]{4}", "", item) for item in x.split() if pattern.match(item)]))))
    
    pickle.dump(tweets_hashtags, open(tweets_hashtags_test_filename, "wb"))    
    
    
    data_df = []; tweets_hashtags = []
    
    
    



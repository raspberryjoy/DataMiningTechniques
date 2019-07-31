#!/usr/bin/env python
# -*- coding: utf-8 -*-


################ IMPORTS

from os import path
import re, sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import cPickle as pickle

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import nltk
import sklearn

from operator import add
from random import randint

import seaborn as sns

from aux import *



################ FUNCTIONS

def hist(data_df, rows_idx, col_idx, color, title, xlabel, output_filename):

    if -1 == rows_idx[0]:
        n, bins, patches = plt.hist(data_df[:,col_idx], 20, density=False, facecolor=color, alpha=0.75)
    else:
        n, bins, patches = plt.hist(data_df[rows_idx,col_idx], 20, density=False, facecolor=color, alpha=0.75)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_filename)                                                                                             
    plt.close() 
    
    
def hist_combined(data_df, pos_rows, neg_rows, neu_rows, col_idx, title_pref, output_pref):

    plt.hist(data_df[pos_rows,col_idx], 20, density=False, facecolor='g', alpha=0.5, label='positive')
    plt.hist(data_df[neg_rows,col_idx], 20, density=False, facecolor='r', alpha=0.5, label='negative')
    plt.hist(data_df[neu_rows,col_idx], 20, density=False, facecolor='tab:orange', alpha=0.5, label='neutral')

    plt.legend(loc='upper right')    
    plt.title('Histogram of ' + title_pref)
    plt.xlabel('')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(img_output_dir + '/hist_' + output_pref + '_combined.png')                                                                                             
    plt.close()  
    
    
def wordcloud(words, title, words_color, output_filename, max_words, is_np):

    if True == is_np:
        wordcloud = WordCloud(background_color='white', max_words=max_words,
                                color_func=lambda *args, **kwargs: words_color).generate(' '.join(list(words)))    
    else:
        wordcloud = WordCloud(background_color='white', max_words=max_words,
                                color_func=lambda *args, **kwargs: words_color).generate(words.to_string())
    
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(title)
    plt.axis('off')
    plt.savefig(output_filename)
    plt.close()     

################ END FUNCTIONS






## Analysis
def analysis():

    train_lemm = pickle.load(open(toklemm_tweets_filename, "rb"))

    neg_rows = np.array(pickle.load(open(x_train_neg_filename,"rb")))
    neu_rows = np.array(pickle.load(open(x_train_neu_filename,"rb")))
    pos_rows = np.array(pickle.load(open(x_train_pos_filename,"rb")))



    # Word cloud

    # Total
    wordcloud(train_lemm, 'Total tweets: words (lemm.)', 'black', img_output_dir + '/wordcloud_lemm_total.png', 50, False)
    # Positive
    wordcloud(train_lemm[pos_rows], 'Positive tweets: words (lemm.)', 'green', img_output_dir + '/wordcloud_lemm_pos.png', 50, False)
    # Negative
    wordcloud(train_lemm[neg_rows], 'Negative tweets: words (lemm.)', 'red', img_output_dir + '/wordcloud_lemm_neg.png', 50, False)
    # Neutral
    wordcloud(train_lemm[neu_rows], 'Neutral tweets: words (lemm.)', 'orange', img_output_dir + '/wordcloud_lemm_neu.png', 50, False)

    train_lemm = []
    
    
    
    train_stem = pickle.load(open(toklemm_tweets_filename, "rb"))
    
    # Total
    wordcloud(train_stem, 'Total tweets: words (stemm.)', 'black', img_output_dir + '/wordcloud_stemm_total.png', 50, False)
    # Positive
    wordcloud(train_stem[pos_rows], 'Positive tweets: words (stemm.)', 'green', img_output_dir + '/wordcloud_stemm_pos.png', 50, False)
    # Negative
    wordcloud(train_stem[neg_rows], 'Negative tweets: words (stemm.)', 'red', img_output_dir + '/wordcloud_stemm_neg.png', 50, False)
    # Neutral
    wordcloud(train_stem[neu_rows], 'Neutral tweets: words (stemm.)', 'orange', img_output_dir + '/wordcloud_stemm_neu.png', 50, False)

    train_stem = []    



    tweets_hashtags = pickle.load(open(tweets_hashtags_filename, "rb"))


    # Hashtags

    # Total
    wordcloud(tweets_hashtags, 'Total tweets: hashtags', 'black', img_output_dir + '/wordcloud_hashtags_total.png', 20, True)
    # Positive
    wordcloud(tweets_hashtags[pos_rows], 'Positive tweets: hashtags', 'green', img_output_dir + '/wordcloud_hashtags_pos.png', 20, True)
    # Negative
    wordcloud(tweets_hashtags[neg_rows], 'Negative tweets: hashtags', 'red', img_output_dir + '/wordcloud_hashtags_neg.png', 20, True)
    # Neutral
    wordcloud(tweets_hashtags[neu_rows], 'Neutral tweets: hashtags', 'orange', img_output_dir + '/wordcloud_hashtags_neu.png', 20, True)

    tweets_hashtags = [] 





    extra_df = np.array(pickle.load(open(extrafeatures_mat_filename, "rb")))
        

    # #words

    title_pref = 'the number of words'
    output_pref = 'words'
    x_title = 'words'
    cur_idx = idx['num_words']

    # Total
    hist(extra_df, [-1], cur_idx, 'k', 'Histogram of ' + title_pref + ' - total', 
         x_title, img_output_dir + '/hist_' + output_pref + '_total.png')
    # Positive
    hist(extra_df, pos_rows, cur_idx, 'g', 'Histogram of ' + title_pref + ' - positive', 
         x_title, img_output_dir + '/hist_' + output_pref + '_pos.png')
    # Negative
    hist(extra_df, neg_rows, cur_idx, 'r', 'Histogram of ' + title_pref + ' - negative', 
         x_title, img_output_dir + '/hist_' + output_pref + '_neg.png')
    # Neutral
    hist(extra_df, neu_rows, cur_idx, 'tab:orange', 'Histogram of ' + title_pref + ' - neutral', 
         x_title, img_output_dir + '/hist_' + output_pref + '_neu.png')
    # Combined
    hist_combined(extra_df, pos_rows, neg_rows, neu_rows, cur_idx, title_pref, output_pref)


    # # chars

    title_pref = 'the tweets\' length'
    output_pref = 'len'
    x_title = 'len'
    cur_idx = idx['tweet_len']

    # Total
    hist(extra_df, [-1], cur_idx, 'k', 'Histogram of ' + title_pref + ' - total', 
         x_title, img_output_dir + '/hist_' + output_pref + '_total.png')
    # Positive
    hist(extra_df, pos_rows, cur_idx, 'g', 'Histogram of ' + title_pref + ' - positive', 
         x_title, img_output_dir + '/hist_' + output_pref + '_pos.png')
    # Negative
    hist(extra_df, neg_rows, cur_idx, 'r', 'Histogram of ' + title_pref + ' - negative',  
         x_title, img_output_dir + '/hist_' + output_pref + '_neg.png')
    # Neutral
    hist(extra_df, neu_rows, cur_idx, 'tab:orange', 'Histogram of ' + title_pref + ' - neutral',  
         x_title, img_output_dir + '/hist_' + output_pref + '_neu.png')
    # Combined
    hist_combined(extra_df, pos_rows, neg_rows, neu_rows, cur_idx, title_pref, output_pref)
    
    
    
    # # vowels

    title_pref = 'the number of vowels'
    output_pref = 'vowels'
    x_title = 'vowels'
    cur_idx = idx['num_vowels']

    # Total
    hist(extra_df, [-1], cur_idx, 'k', 'Histogram of ' + title_pref + ' - total', 
         x_title, img_output_dir + '/hist_' + output_pref + '_total.png')
    # Positive
    hist(extra_df, pos_rows, cur_idx, 'g', 'Histogram of ' + title_pref + ' - positive', 
         x_title, img_output_dir + '/hist_' + output_pref + '_pos.png')
    # Negative
    hist(extra_df, neg_rows, cur_idx, 'r', 'Histogram of ' + title_pref + ' - negative',  
         x_title, img_output_dir + '/hist_' + output_pref + '_neg.png')
    # Neutral
    hist(extra_df, neu_rows, cur_idx, 'tab:orange', 'Histogram of ' + title_pref + ' - neutral',  
         x_title, img_output_dir + '/hist_' + output_pref + '_neu.png')
    # Combined
    hist_combined(extra_df, pos_rows, neg_rows, neu_rows, cur_idx, title_pref, output_pref)    



    # no. emoticons

    title_pref = 'the number of emoticons'
    output_pref = 'emoticons'
    x_title = 'No. emoticons'
    cur_idx = idx['num_hashtags']

    # Total
    hist(extra_df, [-1], cur_idx, 'k', 'Histogram of ' + title_pref + ' - total', 
         x_title, img_output_dir + '/hist_' + output_pref + '_total.png')
    # Positive
    hist(extra_df, pos_rows, cur_idx, 'g', 'Histogram of ' + title_pref + ' - positive',
         x_title, img_output_dir + '/hist_' + output_pref + '_pos.png')
    # Negative
    hist(extra_df, neg_rows, cur_idx, 'r', 'Histogram of ' + title_pref + ' - negative',  
         x_title, img_output_dir + '/hist_' + output_pref + '_neg.png')
    # Neutral
    hist(extra_df, neu_rows, cur_idx, 'tab:orange', 'Histogram of ' + title_pref + ' - neutral',  
         x_title, img_output_dir + '/hist_' + output_pref + '_neu.png')
    # Combined
    hist_combined(extra_df, pos_rows, neg_rows, neu_rows, cur_idx, title_pref, output_pref)
         
         
         
    # no. hashtags

    title_pref = 'the number of hashtags'
    output_pref = 'hashtags'
    x_title = 'No. hashtags'
    cur_idx = idx['num_hashtags']

    # Total
    hist(extra_df, [-1], cur_idx, 'k', 'Histogram of ' + title_pref + ' - total', 
         x_title, img_output_dir + '/hist_' + output_pref + '_total.png')
    # Positive
    hist(extra_df, pos_rows, cur_idx, 'g', 'Histogram of ' + title_pref + ' - positive', 
         x_title, img_output_dir + '/hist_' + output_pref + '_pos.png')
    # Negative
    hist(extra_df, neg_rows, cur_idx, 'r', 'Histogram of ' + title_pref + ' - negative', 
         x_title, img_output_dir + '/hist_' + output_pref + '_neg.png')
    # Neutral
    hist(extra_df, neu_rows, cur_idx, 'tab:orange', 'Histogram of ' + title_pref + ' - neutral',  
         x_title, img_output_dir + '/hist_' + output_pref + '_neu.png')
    # Combined
    hist_combined(extra_df, pos_rows, neg_rows, neu_rows, cur_idx, title_pref, output_pref)



    # no. urls
    title_pref = 'the number of URLs'
    output_pref = 'urls'
    x_title = 'No. URLs'
    cur_idx = idx['num_urls']

    # Total
    hist(extra_df, [-1], cur_idx, 'k', 'Histogram of ' + title_pref + ' - total', 
         x_title, img_output_dir + '/hist_' + output_pref + '_total.png')
    # Positive
    hist(extra_df, pos_rows, cur_idx, 'g', 'Histogram of ' + title_pref + ' - positive', 
         x_title, img_output_dir + '/hist_' + output_pref + '_pos.png')
    # Negative
    hist(extra_df, neg_rows, cur_idx, 'r', 'Histogram of ' + title_pref + ' - negative', 
         x_title, img_output_dir + '/hist_' + output_pref + '_neg.png')
    # Neutral
    hist(extra_df, neu_rows, cur_idx, 'tab:orange', 'Histogram of ' + title_pref + ' - neutral',  
         x_title, img_output_dir + '/hist_' + output_pref + '_neu.png')
    # Combined
    hist_combined(extra_df, pos_rows, neg_rows, neu_rows, cur_idx, title_pref, output_pref)
         


    # no. mentions

    title_pref = 'the number of mentions'
    output_pref = 'mentions'
    x_title = 'No. mentions'
    cur_idx = idx['num_mentions']

    # Total
    hist(extra_df, [-1], cur_idx, 'k', 'Histogram of ' + title_pref + ' - total', 
         x_title, img_output_dir + '/hist_' + output_pref + '_total.png')
    # Positive
    hist(extra_df, pos_rows, cur_idx, 'g', 'Histogram of ' + title_pref + ' - positive', 
         x_title, img_output_dir + '/hist_' + output_pref + '_pos.png')
    # Negative
    hist(extra_df, neg_rows, cur_idx, 'r', 'Histogram of ' + title_pref + ' - negative', 
         x_title, img_output_dir + '/hist_' + output_pref + '_neg.png')
    # Neutral
    hist(extra_df, neu_rows, cur_idx, 'tab:orange', 'Histogram of ' + title_pref + ' - neutral', 
         x_title, img_output_dir + '/hist_' + output_pref + '_neu.png')
    # Combined
    hist_combined(extra_df, pos_rows, neg_rows, neu_rows, cur_idx, title_pref, output_pref)
         



    # no. exclamation marks

    title_pref = 'the number of exclamation marks'
    output_pref = 'excl'
    x_title = 'No. exclamation marks'
    cur_idx = idx['num_excl']

    # Total
    hist(extra_df, [-1], cur_idx, 'k', 'Histogram of ' + title_pref + ' - total', 
         x_title, img_output_dir + '/hist_' + output_pref + '_total.png')
    # Positive
    hist(extra_df, pos_rows, cur_idx, 'g', 'Histogram of ' + title_pref + ' - positive',
         x_title, img_output_dir + '/hist_' + output_pref + '_pos.png')
    # Negative
    hist(extra_df, neg_rows, cur_idx, 'r', 'Histogram of ' + title_pref + ' - negative',  
         x_title, img_output_dir + '/hist_' + output_pref + '_neg.png')
    # Neutral
    hist(extra_df, neu_rows, cur_idx, 'tab:orange', 'Histogram of ' + title_pref + ' - neutral', 
         x_title, img_output_dir + '/hist_' + output_pref + '_neu.png')
    # Combined
    hist_combined(extra_df, pos_rows, neg_rows, neu_rows, cur_idx, title_pref, output_pref)





    # Mean, min., max. valence - lexica

    for cur_idx in range(len(lexica)):
        
        # Mean valence

        title_pref = 'the mean valence ~ dictionary #' + str(cur_idx+1) + '\n(' + lexica[cur_idx] + ')'
        output_pref = 'mean_lex' + str(cur_idx+1)
        x_title = 'Mean valence'

        # Total
        hist(extra_df, [-1], (3 * cur_idx), 'k', 'Histogram of ' + title_pref + ' - total', 
             x_title, img_output_dir + '/hist_' + output_pref + '_total.png')
        # Positive
        hist(extra_df, pos_rows, (3 * cur_idx), 'g', 'Histogram of ' + title_pref + ' - positive',
             x_title, img_output_dir + '/hist_' + output_pref + '_pos.png')
        # Negative
        hist(extra_df, neg_rows, (3 * cur_idx), 'r', 'Histogram of ' + title_pref + ' - negative',  
             x_title, img_output_dir + '/hist_' + output_pref + '_neg.png')
        # Neutral
        hist(extra_df, neu_rows, (3 * cur_idx), 'tab:orange', 'Histogram of ' + title_pref + ' - neutral', 
             x_title, img_output_dir + '/hist_' + output_pref + '_neu.png')
        # Combined
        hist_combined(extra_df, pos_rows, neg_rows, neu_rows, (3 * cur_idx), title_pref, output_pref)
        
        
        
        
        # Min. valence

        title_pref = 'the min. valence ~ dictionary #' + str(cur_idx+1) + '\n(' + lexica[cur_idx] + ')'
        output_pref = 'min_lex' + str(cur_idx+1)
        x_title = 'Min. valence'

        # Total
        hist(extra_df, [-1], (3 * cur_idx), 'k', 'Histogram of ' + title_pref + ' - total', 
             x_title, img_output_dir + '/hist_' + output_pref + '_total.png')
        # Positive
        hist(extra_df, pos_rows, (3 * cur_idx), 'g', 'Histogram of ' + title_pref + ' - positive',
             x_title, img_output_dir + '/hist_' + output_pref + '_pos.png')
        # Negative
        hist(extra_df, neg_rows, (3 * cur_idx), 'r', 'Histogram of ' + title_pref + ' - negative',  
             x_title, img_output_dir + '/hist_' + output_pref + '_neg.png')
        # Neutral
        hist(extra_df, neu_rows, (3 * cur_idx), 'tab:orange', 'Histogram of ' + title_pref + ' - neutral', 
             x_title, img_output_dir + '/hist_' + output_pref + '_neu.png')
        # Combined
        hist_combined(extra_df, pos_rows, neg_rows, neu_rows, (3 * cur_idx), title_pref, output_pref)
        
        
        
        
        # Max. valence

        title_pref = 'the max. valence ~ dictionary #' + str(cur_idx+1) + '\n(' + lexica[cur_idx] + ')'
        output_pref = 'max_lex' + str(cur_idx+1)
        x_title = 'Max. valence'

        # Total
        hist(extra_df, [-1], (3 * cur_idx), 'k', 'Histogram of ' + title_pref + ' - total', 
             x_title, img_output_dir + '/hist_' + output_pref + '_total.png')
        # Positive
        hist(extra_df, pos_rows, (3 * cur_idx), 'g', 'Histogram of ' + title_pref + ' - positive',
             x_title, img_output_dir + '/hist_' + output_pref + '_pos.png')
        # Negative
        hist(extra_df, neg_rows, (3 * cur_idx), 'r', 'Histogram of ' + title_pref + ' - negative',  
             x_title, img_output_dir + '/hist_' + output_pref + '_neg.png')
        # Neutral
        hist(extra_df, neu_rows, (3 * cur_idx), 'tab:orange', 'Histogram of ' + title_pref + ' - neutral', 
             x_title, img_output_dir + '/hist_' + output_pref + '_neu.png')
        # Combined
        hist_combined(extra_df, pos_rows, neg_rows, neu_rows, (3 * cur_idx), title_pref, output_pref)    


    extra_df = []




    # Pie-chart of the classes
    train_df = pickle.load(open(train_df_prep_filename, "rb"))
    pie_data = train_df.groupby('class').size()  

    sns.set()
    pie_data.plot(kind='pie', title='Percentages of the classes', figsize=[6,6],
              autopct=lambda p: '{:.1f}% ({:.0f})'.format(p,(p/100)*pie_data.sum()))
    #plt.show()

    plt.ylabel('')
    plt.savefig(img_output_dir + '/pie_class.png')                                                                                             
    plt.close()                            
            
         
    train_df = []                                                       





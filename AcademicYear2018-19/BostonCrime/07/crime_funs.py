#!/usr/bin/env python
# -*- coding: utf-8 -*-


################ IMPORTS

import os, sys
from os import mkdir, path
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
sns.set(style="whitegrid")
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster
import warnings
warnings.filterwarnings("ignore")


################ FUNCTIONS

def hist(data_df, col, output_filename):

    cur_data = data_df[[col]].copy()

    if data_df[col].dtype != np.int64:
        cur_data[col] = cur_data[col].replace(np.nan, 'Unknown', regex=True) 
    

    sns_plot = sns.countplot(data=cur_data, x=col, order=cur_data[col].value_counts().index)
        
    fig = sns_plot.get_figure()
    fig.autofmt_xdate()
    fig.savefig(output_filename)     
    plt.clf()
    
    
def scatt(data_df, cols, labels, title, output_filename):

    colors = np.array(['xkcd:azure','xkcd:chocolate',
                'xkcd:goldenrod','xkcd:aquamarine',
                'xkcd:coral','xkcd:magenta',
                'xkcd:chartreuse','xkcd:lavender',
                'xkcd:olive','xkcd:pink'])


    # Generate and extra 3D plot
    if 3 == len(cols):
    
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(data_df[cols[0]],
                    data_df[cols[1]],
                    data_df[cols[2]],
                    c=colors[labels],
                    s=0.5)

        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_zlabel(cols[2])    
        ax.set_title(title)
        
        plt.savefig(output_filename[0:(len(output_filename)-4)] + "_3D.png")
        plt.close()
              
        
    
    plt.scatter(data_df[cols[0]],
                data_df[cols[1]], 
                c=colors[labels])

    plt.title(title)
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.grid(True)
        
        
    
    plt.savefig(output_filename)
    plt.close()
    
    
    
def km_clust(data_df, k, cols, img_output_dir):

    cur_data_df = data_df[cols].dropna()

    km = KMeans(n_clusters=k,
               precompute_distances='auto',  
               copy_x=True, 
               algorithm='auto').fit(cur_data_df)

    km.predict(cur_data_df)


    # Create and save the scatterplot
    scatt(cur_data_df, cols, km.labels_, 
            'K-means: ' + str(k) + ' clusters | ' + ', '.join(cols), 
            img_output_dir + '/scatter_kmeans_' + str(k) + '_' + '_'.join(cols) + '.png')    
   
################ END FUNCTIONS


img_output_dir = 'img'



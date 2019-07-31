import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

####################
# Loading the data #
####################
fields = ['MONTH','Lat','Long']
df = pd.read_csv(os.getcwd()+'/crime.csv',skipinitialspace=True, usecols=fields)

######################
# Preparing the data #
######################
df = df.dropna() # deleting missing values
df = df.loc[(df['Lat']>40)&(df['Long']<-60)] # removing outliers

###########################
# Clustering with K-Means #
###########################
usr_input = "not_stop"
while usr_input != "stop":
    print("Please, choose the desired number of clusters",end='')
    print(" or type stop to exit")
    usr_input = input()

    if usr_input != "stop":
        km = KMeans(n_clusters = int(usr_input))
        km.fit(df)
        km.predict(df)
        labels = km.labels_

        #########################
        # Plotting the clusters #
        #########################
        fig = plt.figure(1,figsize = (20,20))
        ax = Axes3D(fig,rect=[0,0,0.95,1],elev=48,azim=134)
        ax.scatter(df['Lat'],df['Long'],df['MONTH'],c=labels.astype(np.float),edgecolor='k',s=50)
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Longitude")
        ax.set_zlabel("Month")
        plt.title("Kmeans with " + usr_input + " clusters")
        plt.show()

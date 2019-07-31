import sys
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# np.set_printoptions(threshold=sys.maxsize)

####################
# Getting the data #
####################
f = open('new_dataset.pickle', 'rb')
df = pkl.load(f)
f.close()

coordinates = df[['Lat','Long']] # Our new dataset
coordinates = coordinates.dropna() # Dropping missing values
coordinates = coordinates.loc[(coordinates['Lat']>40) &
                (coordinates['Long'] < -60)]

##############################
# Plotting the original data #
##############################
# A scatter plot using the seaborn library
ax = sns.scatterplot(x='Long',y='Lat',data=coordinates)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.title("Scatter plot using the seaborn library")
plt.show()

# A scatter plot using matplotlib
colors = np.random.rand(len(coordinates))
plt.figure(figsize=(20,20))
plt.scatter(x = coordinates['Long'], y = coordinates['Lat'], c = colors, alpha = 0.5)
plt.title("Scatter plot using the matplotlib library")
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

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
        km.fit(coordinates)
        y_kmeans = km.predict(coordinates)
        labels = km.labels_
        # print(labels)

        #########################
        # Plotting the clusters #
        #########################
        # Scatter plot of the clusters using matplotlib
        plt.figure(figsize=(20,20))
        plt.scatter(x = coordinates['Long'], y = coordinates['Lat'], c = y_kmeans, alpha = 0.5)
        plt.title("Kmeans with " + usr_input + " clusters")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

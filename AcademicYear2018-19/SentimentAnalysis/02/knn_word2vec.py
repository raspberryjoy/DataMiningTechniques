import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

####################
# Loading the data #
####################

# Getting the train set
train_file_name = "clean_train2017_b"
path = os.getcwd() + '/DataMining/twitter_data/' + train_file_name + ".tsv"
df_train = pd.DataFrame(pd.read_csv(path,sep='\t'))

# Getting the test set
test_file_name = "clean_test2017_b"
path = os.getcwd() + '/DataMining/twitter_data/' + test_file_name + ".tsv"
df_test = pd.DataFrame(pd.read_csv(path,sep='\t'))
# print(df_test)
df_test['Emotion'] = ''
# print(df_test)

pkl_path = os.getcwd() + '/pkl/'
pkl_file = open(pkl_path+train_file_name+'_word2vec_tweet_vectors.pkl', 'rb')
df_tweets_train = pickle.load(pkl_file)
pkl_file.close()

pkl_path = os.getcwd() + '/pkl/'
pkl_file = open(pkl_path+test_file_name+'_word2vec_tweet_vectors.pkl', 'rb')
df_tweets_test = pickle.load(pkl_file)
pkl_file.close()
# print(df_tweets_test)
df_tweets_test['Emotion'] = ''
# print(df_tweets_test)

X = list(df_tweets_train['Tweet Vector'])
y = df_tweets_train['Emotion']

X_train = list(df_tweets_train['Tweet Vector'])
X_test = list(df_tweets_test['Tweet Vector'])
y_train = df_tweets_train['Emotion']
y_test = df_tweets_test['Emotion']

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Making predictions
y_pred = classifier.predict(X_test)
predictions = y_pred.tolist()

i = 0
while i < len(df_tweets_test.index):
    print("[",i,"]")
    # df_tweets_test.iloc[i]['Emotion'] = predictions[i]
    df_test.iloc[i]['Emotion'] = predictions[i]
    i = i + 1
print(df_test)

# Evaluating the algorithm
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

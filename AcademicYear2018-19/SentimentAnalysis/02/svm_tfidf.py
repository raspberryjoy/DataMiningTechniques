import os
import pickle
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

####################
# Loading the data #
####################

# Getting the train set
train_file_name = "clean_train2017_b"
path = os.getcwd() + '/DataMining/twitter_data/' + train_file_name + ".tsv"
df = pd.DataFrame(pd.read_csv(path,sep='\t'))
# print(df)

# Getting the test set
test_file_name = "clean_test2017_b"
path = os.getcwd() + '/DataMining/twitter_data/' + test_file_name + ".tsv"
df_test = pd.DataFrame(pd.read_csv(path,sep='\t'))
# print(df_test)
df_test['Emotion'] = ''
# print(df_test)

# Getting the vectors that occured from the Bag of Words method
pkl_path = os.getcwd() + '/pkl/'
pkl_file = open(pkl_path+train_file_name+'_tfidf.pkl', 'rb')
vector_train = pickle.load(pkl_file)
pkl_file.close()
# print(vector_train.toarray())
# print(type(vector_train))

pkl_path = os.getcwd() + '/pkl/'
pkl_file = open(pkl_path+test_file_name+'_tfidf.pkl', 'rb')
vector_test = pickle.load(pkl_file)
pkl_file.close()
# print(vector_test.toarray())
# print(type(vector_test))

X = vector_train # The vector which occured from the Bag of Words method
y = df['Emotion']

X_train = vector_train
X_test = vector_test
y_train = df['Emotion']
y_test = df_test['Emotion']

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

# Making predictions
y_pred = svclassifier.predict(X_test)

predictions = y_pred.tolist()

i = 0
while i < len(df_test.index):
    print("[",i,"]")
    df_test.iloc[i]['Emotion'] = y_pred[i]
    i = i + 1
print(df_test)

# Evaluating the algorithm
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

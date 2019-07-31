import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def main():

    #################
    # KNN for TFIDF #
    #################

    ####################
    # Loading the data #
    ####################
    path = os.getcwd() + '/DataMining/twitter_data/clean_train2017_c.tsv'
    df = pd.DataFrame(pd.read_csv(path,sep='\t'))
    # print(df)

    pkl_file = open('tfidf.pkl', 'rb')
    vector = pickle.load(pkl_file)
    pkl_file.close()
    # print(vector.toarray())
    # print(type(vector))

    X = vector
    y = df['Emotion']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    # Making predictions
    y_pred = classifier.predict(X_test)

    # Evaluating the algorithm
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()

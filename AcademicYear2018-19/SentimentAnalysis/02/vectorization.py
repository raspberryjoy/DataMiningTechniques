import os
import ast
import pandas as pd
import pickle as pkl

from functions import prepare_data, word2vector
from sklearn.feature_extraction.text import CountVectorizer # Bag of Words
from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF

# def main():

####################
# Getting the Data #
####################
print("Welcome! Please, enter the name of the clean dataset you'll work with.")
print("*** WITHOUT THE .tsv ***")
file_name = input( )

# Setting up the file's path
dirpath = os.getcwd() # Current working directory
dataSetPath = dirpath + '/DataMining/twitter_data/' + file_name + ".tsv" # File path
# print(dataSetPath)

# Reading the data
df = pd.DataFrame(data=pd.read_csv(dataSetPath, sep='\t'))
# print("\nOriginal Data:\n",df)

########################################
#            Vectorization             #
########################################

print("Choose one of the following vectorization methods: ")
print("* Bag Of Words\n* TF-IDF\n* Word Embeddings\n")
method = input()

# The vectorizer we will be using
if method == "Word Embeddings":
    word2vector(df,file_name)
elif method == "Bag Of Words" or method == "TF-IDF":
    if method == "Bag Of Words":
        vectorizer = CountVectorizer(max_features=1000)
    else:
        vectorizer = TfidfVectorizer(max_features=1000)

    corpus = prepare_data(df)

    # To actually create the vectorizer, we simply need to call fit on the text
    # Learn the vocabulary dictionary and return term-document matrix.
    vector = vectorizer.fit_transform(corpus)

    # vectorizer.vocabulary_ is a dictionary. The number of elements it holds is
    # the same as the total number of words in our vocabulary. Each key repre-
    # sents a word from the vocabulary and its value is that word's index in
    # the vector.
    # print(vectorizer.vocabulary_,"\n")

    # Array mapping from feature integer indices to feature name
    # print(vectorizer.get_feature_names())

    pkl_path = os.getcwd() + '/pkl/' # the directory where the .pkl will be stored

    if method == "Bag Of Words":
        # print(vector.toarray())
        # print(vector)
        output = open(pkl_path+file_name+'_bow.pkl','wb')
        pkl.dump(vector ,output)
        output.close()
    elif method == "TF-IDF":
        # print(vector)
        output = open(pkl_path+file_name+'_tfidf.pkl','wb')
        pkl.dump(vector,output)
        output.close()

# if __name__ == '__main__':
#     main()

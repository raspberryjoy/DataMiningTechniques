import os
import re
import ast
import numpy as np
import pandas as pd
import pickle as pkl

# Here, we create a new vector for each tweet.
# The vector is created by adding up the vectors of every token and then diving
# the result by the number of token the tweet has.
# When we're done we store the new vectors, along with a unique ID,
# to a DataFrame
# def main():

####################
# Loading the data #
####################
print("Please, enter the name of the clean data set you're working with")
print("*** WITHOUT THE .tsv ***")
file_name = input()

# The cleaned data
path = os.getcwd() + '/DataMining/twitter_data/' + file_name + ".tsv"
df = pd.DataFrame(data=pd.read_csv(path,sep='\t'))

# The vectors including the ones with added features
pkl_path = os.getcwd() + '/pkl/'
a = open(pkl_path+file_name+'_renewed_vectors.pkl',"rb")
vectors = pkl.load(a)

# A dictionary for finding the vector of every word
b = open(pkl_path+file_name+'_vector_indexes.pkl',"rb")
dict = pkl.load(b)

df_new = pd.DataFrame(columns=['Tweet ID','Emotion','Tweet Vector']) # where we will store the data

i = 0
while i < len(df.index): # traversing through the tweets
    print("[",i,"]")
    tmp_list = [] # the vector for the tweet
    for token in ast.literal_eval(df.iloc[i]['Tweet']): # traversing through the tokens of the current tweet
        if token in dict: # there is a vector for the current token
            tmp_list.append(vectors[dict[token]])
        else: # token doesn't have a vector
            tmp_list.append(np.random.rand(301,)) # create a random vector

    # tmp_list is now a list of arrays. Each array represents the vector
    # of a word (token)
    tweet_vector = tmp_list[0]
    j = 1
    while j < len(tmp_list):
        tweet_vector = tweet_vector + tmp_list[j]
        j = j + 1
    tweet_vector = tweet_vector/len(ast.literal_eval(df.iloc[i]['Tweet']))

    df_new.loc[i] = [i,df.iloc[i]['Emotion'],tweet_vector]
    i = i + 1

# print(df_new)

# Saving to a .pkl file
output = open(pkl_path+file_name+'_word2vec_tweet_vectors.pkl','wb')
pkl.dump(df_new,output)
output.close()

# if __name__ == '__main__':
#     main()

import os
import re
import sys
import nltk
import string
import pandas as pd
from functions import clean_tweets

# Display options
pd.set_option('display.max_colwidth', -1)

print("Welcome! Please, enter the name of the dataset that will be cleaned")
file_name = input( )

####################
# Getting the Data #
####################

# Setting up the file's path
dirpath = os.getcwd() # Current working directory
dataSetPath = dirpath + '/DataMining/twitter_data/' + file_name # File path
# print(dataSetPath)

# Reading the data
df = pd.DataFrame(data=pd.read_csv(dataSetPath, header = None, sep='\t',
    names=['Code1','Code2','Emotion','Tweet']))
# print("\nOriginal Data:\n",df)

##################################
# Modifying the DataFrame object #
##################################

# Converting text to lowercase
df['Tweet'] = df['Tweet'].str.lower()
# print("\nData in lowercase:\n",df)

# Removing all duplicates
df.drop_duplicates(keep='first',inplace=True)
# print("\nData after the removal of duplicates:\n",df)

# Fixing the indexes
df.reset_index(drop=True, inplace=True)
# print("\nAfter reseting the indexes:\n",df)

# Deleting the first two columns since they don't have useful information
del df['Code1']
del df['Code2']
# print("\nAfter deleting one column:\n",df)

#######################
# Cleaning the tweets #
#######################
indexes_to_drop = []
indexes_to_drop = clean_tweets(df)

df = df.drop(indexes_to_drop)
df.reset_index(drop=True, inplace=True)
# print("\nAfter cleaning the tweets:\n",df)

######################
# Exporting the data #
######################
path = dirpath + '/DataMining/twitter_data/clean_' + file_name
df.to_csv(path,sep='\t',index=None)

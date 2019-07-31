import os
import numpy
import pickle
import gensim
import numpy as np

from numpy import array
from functions import txt_to_dict
from gensim.models import Word2Vec

# def main():

#####################
# Loading the model #
#####################
print("Please, enter the name of the clean data set you're working with")
print("*** WITHOUT THE .tsv ***")
file_name = input()

pkl_path = os.getcwd() + '/pkl/'
pickle_in = open(pkl_path+file_name+'_word2vec.pkl',"rb")
w2v_model = pickle.load(pickle_in)

########################################
# Getting the sentimental dictionaries #
########################################
affin = {}
affin = txt_to_dict("/affin/affin.txt")
# for key in affin.keys():
#     print(key)

emotweet = {}
emotweet = txt_to_dict("/emotweet/valence_tweet.txt")
# for key in emotweet.keys():
#     print(key,":",type(emotweet[key]))

generic = {}
generic = txt_to_dict("/generic/generic.txt")
# for key in generic.keys():
#     print(key)

nrc = {}
nrc = txt_to_dict("/nrc/val.txt")
# for key in nrc.keys():
#     print(key)

nrctag = {}
nrctag = txt_to_dict("/nrctag/val.txt")
# for key in nrctag.keys():
#     print(key)

##############################
# Adding features to vectors #
##############################

dict = {} # Connects words to their vectors
vectors = [] # A list of arrays (vectors)
index = 0 # The index of the vector in the list

for word in w2v_model.wv.vocab:
    dictionaries_used = 0
    sum = 0
    avg = 0

    if word in affin:
        sum = sum + affin[word]
        dictionaries_used = dictionaries_used + 1
    if word in emotweet:
        sum = sum + emotweet[word]
        dictionaries_used = dictionaries_used + 1
    if word in generic:
        sum = sum + generic[word]
        dictionaries_used = dictionaries_used + 1
    if word in nrc:
        sum = sum + nrc[word]
        dictionaries_used = dictionaries_used + 1
    if word in nrctag:
        sum = sum + nrctag[word]
        dictionaries_used = dictionaries_used + 1

    if dictionaries_used != 0: # Added a feature
        avg = sum/dictionaries_used

    tmp_array = np.append(w2v_model[word],avg)
    vectors.append(tmp_array) # Adding the vector to the list
    dict[word] = index # Storing the vector's index
    index = index + 1

########################
# Saving to .pkl files #
########################
output = open(pkl_path+file_name+'_renewed_vectors.pkl','wb')
pickle.dump(vectors,output)
output.close()

output = open(pkl_path+file_name+'_vector_indexes.pkl','wb')
pickle.dump(dict,output)
output.close()

# if __name__ == '__main__':
#     main()

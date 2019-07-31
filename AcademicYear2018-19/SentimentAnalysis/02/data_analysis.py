import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import path
from PIL import Image
from collections import Counter
from functions import analyze_data
from wordcloud import WordCloud, ImageColorGenerator

# Display options
# pd.options.display.max_colwidth = -1

# def main():

###############################
# Retrieving the user's input #
###############################
print("Welcome! Please, enter the name of the clean dataset you wish to analyze")
print("(e.g: clean_foo.tsv)")
file = input() # Name of the .tsv file

print("Please, enter an indication [positive/negative/neutral/all]")
indication = input()

# Creating the path
dirpath = os.getcwd() # Current working directory
dirpath = dirpath + '/DataMining/twitter_data/' + file

####################
# Getting the data #
####################
df = pd.DataFrame(data=pd.read_csv(dirpath,sep='\t'))

######################
# Analyzing the data #
######################
comment_words = analyze_data(indication,df)

##########################
# Creating the WordCloud #
##########################
d = os.getcwd() # Current working directory
cloud_mask = np.array(Image.open(path.join(d, "stormtrooper_mask.png")))

wordcloud = WordCloud(width = 800, height = 800, background_color ='white',
            min_font_size = 10,mask=cloud_mask).generate(comment_words)

# Plotting the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show() # Display the worldcloud
wordcloud_path = os.getcwd() + '/WordClouds/'
wordcloud.to_file(indication+'_wordcloud.png') # Save the wordcloud
#
# if __name__ == '__main__':
#     main()

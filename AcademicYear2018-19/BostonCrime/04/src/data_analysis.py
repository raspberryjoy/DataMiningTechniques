import os
import gmplot
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap

#%matplotlib inline

# Display Options
pd.options.display.max_columns = 300
pd.options.display.max_rows = 327820

####################
# Reading the data #
####################
file_path = os.getcwd() + '/crime.csv'
df = pd.read_csv(file_path,encoding="iso-8859-1",low_memory=False)

######################
# Exploring the data #
######################

# print(df.dtypes,"\n")
# print(df.isnull().sum()/df.shape[0],"\n")
# print(df.columns.values)

######################
# Preparing the data #
######################

df['SHOOTING'] = df['SHOOTING'].fillna('N') # filling missing values with 'N's

########################
# Researching the data #
########################
sns.set(style='ticks',palette='Set2')
plt.figure(figsize=(16,8))

# Year
df.groupby(['YEAR'])['INCIDENT_NUMBER'].count().plot(kind='bar')
plt.ylabel('Number of crimes')
plt.xlabel('Year')
plt.title('Crimes reported in the Boston area from 2015 to 2018 ')
plt.show()

# Month
df.groupby(['MONTH'])['INCIDENT_NUMBER'].count().plot(kind='bar')
plt.ylabel('Number of crimes')
plt.xlabel('Month')
plt.title("Crimes reported in the Boston area for each month")
plt.show()

# Day of the week
df.groupby(['DAY_OF_WEEK'])['INCIDENT_NUMBER'].count().plot(kind='bar')
plt.ylabel('Number of crimes')
plt.xlabel('Day of the week')
plt.title('Crimes reported in the Boston area for each day of the week')
plt.show()

# District
df.groupby(['DISTRICT'])['INCIDENT_NUMBER'].count().plot(kind='bar')
plt.ylabel('Number of crimes')
plt.xlabel('District')
plt.title('Crimes reported in the Boston area for each district')
plt.show()

# Shootings per year
df['YEAR'].loc[df['SHOOTING']=='Y'].value_counts().plot.bar()
plt.ylabel('Number of shootings')
plt.xlabel('Year')
plt.title('Shootings reported in the Boston area from 2015 to 2018')
plt.show()

df['DISTRICT'].loc[df['SHOOTING']=='Y'].value_counts().plot.bar()
plt.ylabel('Number of shootings')
plt.xlabel('Year')
plt.title("Shootings reported in Boston's districts")
plt.show()

########################
# Changing the dataset #
########################

# day = [7,8,9,10,11,12,13,14,15,16,17]
# night = [18,19,20,21,22,23,0,1,2,3,4,5,6]
# my_list = []
#
# i = 0
# while i < len(df.index):
#     print("[",i,"]")
#     if df.loc[i]['HOUR'] in day:
#          my_list.append('DAY')
#     elif df.loc[i]['HOUR'] in night:
#         my_list.append('NIGHT')
#     i = i + 1
#
# df['DAY_OR_NIGHT'] = np.array(my_list)
#
# # Save to a pickle file
# with open('new_dataset.pickle', 'wb') as f:
#     pkl.dump(df, f)

# Load from a pickle file
with open('new_dataset.pickle', 'rb') as f:
    df_new = pkl.load(f)
    f.close()

# Crimes during the day and at night
df_new.groupby(['DAY_OR_NIGHT'])['INCIDENT_NUMBER'].count().plot.bar()
plt.ylabel('Number of crimes')
plt.xlabel('Time of the day')
plt.title('Crimes reported in the Boston during the day and at night ')
plt.show()

# Crimes commited during the day
df_new['OFFENSE_CODE_GROUP'].loc[df_new['DAY_OR_NIGHT']=='DAY'].value_counts().plot.bar()
plt.ylabel('Number of crimes')
plt.xlabel('Type of crime')
plt.title("Numbers of different types of crimes commited in the Boston area druing the day")
plt.show()

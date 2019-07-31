import ast
import folium
import pandas as pd
import pickle as pkl

from folium.plugins import MarkerCluster

####################
# Getting the data #
####################
f = open('new_dataset.pickle', 'rb')
df = pkl.load(f)
f.close()

# Only keeping rows with OFFENSE_CODE_GROUP == "Drug Violation"
df = df[df.OFFENSE_CODE_GROUP == "Drug Violation"]
df = df.reset_index()

coordinates = df[['OFFENSE_CODE_GROUP','Lat','Long','Location']]
coordinates = coordinates.loc[(coordinates['Lat']>40) &
                (coordinates['Long'] < -60)]
coordinates = coordinates[['OFFENSE_CODE_GROUP','Location']]

# Turning string representations of tuples into actual tuples
i = 0
while i < len(coordinates.index):
    coordinates.iloc[i]['Location'] =ast.literal_eval(coordinates.iloc[i]['Location'])
    i = i + 1

# Generating a map of Boston
m = folium.Map(location=[42.35843, -71.05977], zoom_start=10)

##############################
# Adding a marker to the map #
##############################
popup = "Boston"
boston_marker = folium.Marker([42.35843, -71.05977], popup=popup)
m.add_child(boston_marker)

##############################
# Generating marker clusters #
##############################
locations = coordinates['Location'].tolist()
icons = [folium.Icon(icon="car", prefix="fa") for _ in range(len(locations))]

cluster = MarkerCluster(locations=locations, icons=icons)
m.add_child(cluster)

##########
# Saving #
##########
m.save('index.html') # Saving the map in a file


# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import folium

df = pd.read_csv('cleaned_df.csv')
df_map = df[['latitude','longitude','price','accommodates']]
df_map.to_csv('df_map.csv')


# In[2]:


from folium.plugins import HeatMap

map_hooray = folium.Map(location = [40.7587,-73.9787],zoom_start = 15,width = '100%')

folium.Marker([40.7794,-73.9632],
            popup = 'The Metropolitan Museum of Art',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)
folium.Marker([40.7589,-73.9851],
            popup = 'Time Square',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)
folium.Marker([40.7484,-73.9857],
            popup = 'Empire State Building',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)
folium.Marker([40.6892,-74.0445],
            popup = 'Statue of Liberty National Monument',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)
folium.Marker([40.7587,-73.9787],
            popup = 'Rockefeller Center',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)
folium.Marker([40.7061,-73.9969],
            popup = 'Brooklyn Bridge',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)
folium.Marker([40.7527,-73.9772],
            popup = 'Grand Central Terminal',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)
folium.Marker([40.7505,-73.9934],
            popup = 'Madison Square Garden',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)
folium.Marker([40.7336,-74.0027],
            popup = 'Greenwich Village',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)
folium.Marker([40.7127,-74.0134],
            popup = 'World Trade Center',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)
folium.Marker([40.7359,-73.9911],
            popup = 'Union Square',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)
folium.Marker([40.7633,-73.9832],
            popup = 'Broadway Theatre',
            icon = folium.Icon(color = 'green')).add_to(map_hooray)

df_map['latitude'] = df_map['latitude'].astype(float)
df_map['longitude'] = df_map['longitude'].astype(float)

heat_data = [[row['latitude'],row['longitude']] for index,row in df_map.iterrows()]

HeatMap(heat_data).add_to(map_hooray)

map_hooray.save('heatmap.html')


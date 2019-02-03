
# coding: utf-8

# In[180]:


import pandas as pd
import numpy as np
import folium


# In[181]:


df = pd.read_csv('cleaned_df.csv')
df_map = df[['latitude','longitude','price','accommodates']]
df_map.to_csv('df_map.csv')


# In[182]:


color_code = ["#736ff7","#6faaf7","#6fe9f7","#6ff7c0","#6ff795","#85f76f","#b7f76f","#ecf76f","#f7ae6f","#f76f6f"]


# In[183]:


def plot_maps(locations):
    # generate a new map
    folium_map = folium.Map(location=[40.738,-73.98],
                           zoom_start=14,
                           width = '100%')
    
    folium.Marker([40.7794,-73.9632],
            popup = 'The Metropolitan Museum of Art',
            icon = folium.Icon(color = 'red')).add_to(folium_map)

    folium.Marker([40.7589,-73.9851],
            popup = 'Time Square',
            icon = folium.Icon(color = 'red')).add_to(folium_map)

    folium.Marker([40.7484,-73.9857],
            popup = 'Empire State Building',
            icon = folium.Icon(color = 'red')).add_to(folium_map)

    folium.Marker([40.6892,-74.0445],
            popup = 'Statue of Liberty National Monument',
            icon = folium.Icon(color = 'red')).add_to(folium_map)

    folium.Marker([40.7587,-73.9787],
            popup = 'Rockefeller Center',
            icon = folium.Icon(color = 'red')).add_to(folium_map)

    folium.Marker([40.7061,-73.9969],
            popup = 'Brooklyn Bridge',
            icon = folium.Icon(color = 'red')).add_to(folium_map)

    folium.Marker([40.7527,-73.9772],
            popup = 'Grand Central Terminal',
            icon = folium.Icon(color = 'red')).add_to(folium_map)

    folium.Marker([40.7505,-73.9934],
            popup = 'Madison Square Garden',
            icon = folium.Icon(color = 'red')).add_to(folium_map)

    folium.Marker([40.7336,-74.0027],
            popup = 'Greenwich Village',
            icon = folium.Icon(color = 'red')).add_to(folium_map)

    folium.Marker([40.7127,-74.0134],
            popup = 'World Trade Center',
            icon = folium.Icon(color = 'red')).add_to(folium_map)

    folium.Marker([40.7359,-73.9911],
            popup = 'Union Square',
            icon = folium.Icon(color = 'red')).add_to(folium_map)

    folium.Marker([40.7633,-73.9832],
            popup = 'Broadway Theatre',
            icon = folium.Icon(color = 'red')).add_to(folium_map)
    
    # for each row in the data, add a circle marker
    for index, row in locations.iterrows():
        size_of_room = row['accommodates']
        price_of_room = row['price']
        
        popup_text = "maximum accommodates: {}<br> price: {}<br>"
        popup_text = popup_text.format(row['accommodates'],
                        row['price'])
        
        color_index = int(price_of_room/100)
        color = color_code[color_index]
    
        folium.CircleMarker(location=(row['latitude'],
                                     row['longitude']),
                            popup=popup_text,
                            radius = size_of_room,
                            fill_opacity = 0.8,
                            color = color,
                            fill  = True).add_to(folium_map)
    legend_html = '''
     <div class = 'my-legend'>
     <div class = 'legend-title'>Price Range</div>
     <div class = 'legend-scale'>
        <ul class = 'legend-labels'>
            <li><span style = "background:#736ff7;"></span>100</li>
            <li><span style = "background:#6faaf7;"></span>200</li>
            <li><span style = "background:#6fe9f7;"></span>300</li>
            <li><span style = "background:#6ff7c0;"></span>400</li>
            <li><span style = "background:#6ff795;"></span>500</li>
            <li><span style = "background:#85f76f;"></span>600</li>
            <li><span style = "background:#b7f76f;"></span>700</li>
            <li><span style = "background:#ecf76f;"></span>800</li>
            <li><span style = "background:#f7ae6f;"></span>900</li>
            <li><span style = "background:#f76f6f;"></span>999</li>        
        </ul>
     </div>
     </div> 
     
    <style type='text/css'>
      .my-legend .legend-title {
        text-align: left;
        margin-bottom: 8px;
        font-weight: bold;
        font-size: 90%;
        }
      .my-legend .legend-scale ul {
        margin: 0;
        padding: 0;
        float: left;
        list-style: none;
        }
      .my-legend .legend-scale ul li {
        display: block;
        float: left;
        width: 50px;
        margin-bottom: 6px;
        text-align: right;
        font-size: 80%;
        list-style: none;
        }
      .my-legend ul.legend-labels li span {
        display: block;
        float: left;
        height: 15px;
        width: 50px;
        }
      .my-legend .legend-source {
        font-size: 70%;
        color: #999;
        clear: both;
        }
      .my-legend a {
        color: #777;
        }
    </style>
      '''
    folium_map.get_root().html.add_child(folium.Element(legend_html))
    return folium_map


# In[184]:


folium_map = plot_maps(df_map)


# In[ ]:


folium_map.save("location_analysis.html")


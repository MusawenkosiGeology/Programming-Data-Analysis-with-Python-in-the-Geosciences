#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd

df = pd.read_excel (r'C:\Users\698422\Desktop\Teaching\Programming\UTM Test Coordinates.xlsx')

print (df)


# In[59]:


df.head()


# In[36]:


df.shape


# In[60]:


df.head(19)


# In[38]:


pip install folium


# In[42]:


# For plotting maps
import folium
import wget
import pandas as pd


# In[40]:


# For plotting in python
import matplotlib
import matplotlib.pyplot as plt


# In[41]:


get_ipython().system('pip3 install wget')


# In[46]:


# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon


# In[44]:


# Start location is HB 5
HB_5_coordinate = [-28.6889575869991, 17.8733631585599]
site_map = folium.Map(location=HB_5_coordinate, zoom_start=10)


# In[48]:


# Create a blue circle at HB 5 coordinate with a popup label showing its name
circle = folium.Circle(HB_5_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('HB 5'))
# Create a blue circle at HB 5 coordinate with a icon showing its name
marker = folium.map.Marker(
    HB_5_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'HB 5',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)


# In[49]:


marker_cluster = MarkerCluster()


# In[61]:


# add marker one by one on the map
for i in range(0,len(df)):
   folium.Marker(
      location=[df.iloc[i]['East'], df.iloc[i]['North']],
      popup=df.iloc[i]['Samples'],
   ).add_to(site_map)

# Show the map again
site_map


# In[ ]:





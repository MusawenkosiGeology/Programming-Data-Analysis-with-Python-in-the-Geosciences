#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


# In[6]:


import pandas as pd 


# In[2]:


pip install python-ternary


# # Load Data

# In[8]:


df_samples = pd.read_excel (r'C:\Users\698422\Desktop\Teaching\Programming\Working Dataset\Basalt_TraceElement_ForStudents.xlsx', sheet_name='Major_Elements_Transpose')
print (df_samples)


# In[4]:


df_samples.head()


# In[9]:


df_samples.tail()


# In[10]:


df_samples.describe()


# In[6]:


df_normalize=df_samples


# In[7]:


sum_column = df_samples["Na2O"] + df_samples["K2O"]
df_samples["Na2O + K2O"] = sum_column
print(df_samples)
df_samples.head()


# In[5]:


SiO2=50.0
if SiO2 >50.0:
    print('SiO2 is greater than 50.0')
elif SiO2 ==50.0
    print('SiO2 is equal to 50.0')
else:
    print('SiO2 is less than 50.0')


# In[8]:


fig,ax = plt.subplots()
sns.scatterplot(data=df_samples, hue='Geological Setting', x='SiO2', y='Na2O + K2O')
#place legend outside top right corner of plot
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Geological Setting')
plt.title('Total alkalis vs. silica (TAS) diagram')
plt.xlabel('SiO2 (wt%)')
plt.ylabel('Na2O + K2O (wt%)')
plt.savefig('scatter.png')
plt.show()


# In[9]:


df_samples.shape


# In[10]:


df_samples.dtypes


# In[11]:


df_samples.groupby('Geological Setting').count()


# In[12]:


data = df_samples.drop(labels=["LOI", "Fe2O3"], axis=1)
data.head()


# # Normalize the data

# The data need to be normalized for a ternary plot. We'll normalize to 100:

# In[13]:


cols = ['Na2O + K2O','FeOT', 'MgO']

for col in cols:
    df_samples[col[0]] = df_samples[col] * 100 / df_samples[cols].sum(axis=1)


# In[14]:


import ternary


# In[15]:


df_samples.head()


# In[16]:


# Set up the figure.
fig, tax = ternary.figure(scale=100)
fig.set_size_inches(10, 9)

# Plot points.
tax.scatter(df_samples[['F', 'M', 'N']].values)

# Plot the points in groups.
for name, group in df_samples.groupby('Geological Setting'):
    
    # Note that we have to shuffle the order.
    # This will place Q at the top, F on the left.
    # So the column order is: right, top, left.
    points = group[['F', 'M', 'N']].values
    tax.scatter(points, marker='o', label=name)
    
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Geological Setting')

# Corner labels.
fontsize = 15
offset = 0.1
tax.top_corner_label("FeO Total (wt.%)", fontsize=fontsize, offset=0.2)
tax.left_corner_label("alkalis (wt.%)", fontsize=fontsize, offset=offset)
tax.right_corner_label("MgO (wt.%)", fontsize=fontsize, offset=offset)

# Decoration.
tax.boundary(linewidth=1)
tax.gridlines(multiple=10, color="gray")
tax.ticks(axis='lbr', linewidth=1, multiple=20)
tax.get_axes().axis('off')

tax.show()


# # Trace Element Data

# In[17]:


df_trace = pd.read_excel (r'C:\Users\698422\Desktop\Teaching\Programming\Working Dataset\Basalt_TraceElement_ForStudents.xlsx', sheet_name='Trace_Element')
print (df_trace)


# In[18]:


df_trace.head()


# In[19]:


data = df_trace.drop(labels=[0,1,2], axis=0)
data


# In[20]:


data["HAWAIIAN_normalized"] = data["HAWAIIAN"]/data["Normalization Data"]
print(data)


# # Histogram plots

# In[24]:


sns.histplot(data=df_samples, x="FeOT", hue="Geological Setting")


# In[26]:


sns.histplot(data=df_samples, x="FeOT", hue="Geological Setting", multiple="stack")


# # Regression Plots

# In[27]:


#Linear Regression Model
from sklearn import linear_model


# In[28]:


X = df_samples[['SiO2']]
y = df_samples['Na2O + K2O']

regr = linear_model.LinearRegression()
regr.fit(X, y)


# In[29]:


features = ['SiO2']
target = 'Na2O + K2O'


# In[30]:


silica = regr.predict([[49.00]])


# In[31]:


print(silica)


# In[33]:


X = df_samples[features].values.reshape(-1, len(features))
y = df_samples[target].values


# In[34]:


ols = linear_model.LinearRegression()
model = ols.fit(X, y)


# In[35]:


print('Features                :  %s' % features)
print('Regression Coefficients : ', [round(item, 2) for item in model.coef_])
print('R-squared               :  %.2f' % model.score(X, y))
print('Y-intercept             :  %.2f' % model.intercept_)
print('')


# In[ ]:





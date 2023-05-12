from __future__ import division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import os
# for dirname, _, filenames in os.walk('nyc_airbnb'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline

# import warnings
# warnings.filterwarnings('ignore')
# import geopandas as gpd #pip install geopandas

# from sklearn import preprocessing
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics

# sns.set_style('darkgrid')

""""
how many rows and columns are there?
Is the data numeric? What are the names of the features (columns)? 
Are there any missing values, text, and numeric symbols inappropriate to the data?
"""
data = pd.read_csv('nyc_airbnb/AB_NYC_2019.csv')
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns/features: {data.shape[1]}")
print(f"List of columns: {data.columns}")
print(f"First and last 5 rows:{pd.concat([data.head(),data.tail()])}")
print(f"Statistical summary of dataset:{data.describe()}")


#top_3_hosts (hosts with the most listings)
top_3_hosts = pd.DataFrame(data.value_counts("host_id")[:3])
top_3_hosts.columns=['Listing']
print(top_3_hosts)
top_3_hosts['host_id'] = top_3_hosts.index  # creates a new column called host_id and assigns index values to the new column
top_3_hosts.reset_index(drop=True, inplace=True) # resets index, drops old index, and changes/operations will occur directly/inplace instead of creating a new copy of the dataframe
print(top_3_hosts)


#top_3_neighborhood_groups (neighborhoods with the most listings)
top_3_neigh = pd.DataFrame(data['neighbourhood_group'].value_counts()[:3])
top_3_neigh.columns=['Listings']
top_3_neigh['neighbourhood_group']=top_3_neigh.index
top_3_neigh.reset_index(drop=True, inplace=True)
print(top_3_neigh)
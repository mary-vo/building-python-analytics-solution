from __future__ import division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns

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

#wordcloud
from wordcloud import WordCloud, ImageColorGenerator
wordcloud = WordCloud(
                          background_color='white'
                         ).generate(" ".join(data['neighbourhood'])) # " ".join is joining all values in the column, separated by a space character
plt.figure(figsize=(15,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('neighbourhood.png')
# plt.show()

# Data cleaning
data.drop(['id','host_id','host_name','last_review'],axis=1,inplace=True)
# print(data)

print(data.isnull().sum())


"""For example, say we want to determine the income of a state, which is not distributed uniformly. A handful of people earning significantly more than the average will produce outliers("lies outside") in the dataset. Outliers are a severe threat to any data analysis. In such cases, the median income will be closer than the mean to the middle-class (majority) income.
Means are handy when data is uniformly distributed."""
# These do the same thing:
data_check_distrib=data.drop(data[pd.isnull(data.reviews_per_month)].index)
# data_check_distrib=data.drop(data[data['reviews_per_month'].isnull()].index)
print(data_check_distrib)

print({"Mean":np.nanmean(data.reviews_per_month),"Median":np.nanmedian(data.reviews_per_month),
 "Standard Dev":np.nanstd(data.reviews_per_month)})


# plot a histogram 
plt.figure(figsize=(15,10))
plt.hist(data_check_distrib.reviews_per_month,  bins=50)
plt.title("Distribution of reviews_per_month")
plt.xlim((min(data_check_distrib.reviews_per_month), max(data_check_distrib.reviews_per_month)))
plt.savefig('neighbourhood.png')
plt.show()

# It is right-skewed! Let's fill the values.
def impute_median(series):
    return series.fillna(series.median())
print(data.reviews_per_month)
data.reviews_per_month=data["reviews_per_month"].transform(impute_median)
print(f"after transform: {data.reviews_per_month}")

# data_check_distrib=data.drop(data[pd.isnull(data.reviews_per_month)].index)
# data_check_distrib=data.drop(data[data['reviews_per_month'].isnull()].index)
# print(data_check_distrib)



# plot a histogram 
plt.figure(figsize=(10,5))
# binwidth = .5
plt.hist(data.reviews_per_month,  bins=[.5,1,1.5,2,2.5,3,3.5,4])
plt.title("Distribution of reviews_per_month, fillna")
# plt.xlim((min(data_check_distrib.reviews_per_month), max(data_check_distrib.reviews_per_month)))
plt.savefig('neighbourhood.png')
plt.show()


# Correlation matrix plot
data['reviews_per_month'].fillna(value=0, inplace=True)
# print(data.columns)
data.drop(columns=['name', 'neighbourhood_group',
       'neighbourhood', 'room_type'],axis=1,inplace=True)
# print(data.columns)
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,linewidths=5,fmt='.1f',ax=ax, cmap='Reds')
plt.show()
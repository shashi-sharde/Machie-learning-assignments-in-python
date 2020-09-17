# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:03:30 2020

@author: shashi
"""


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

users = pd.read_csv('D:/BX-Users.csv', error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1')
users.shape
books = pd.read_csv('D:/BX-Books.csv', error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1')
books.shape
ratings = pd.read_csv('D:/BX-Book-Ratings.csv', error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1')
ratings.shape
#checking for columns
users.columns
books.columns
ratings.columns

#merging ratings and users on basis of user id
data = pd.merge(ratings, users, on='User-ID', how='inner')

#merging data and books on basis of ISBN
data = pd.merge(data, books, on='ISBN', how='inner')
# now all three data set is merged into one 
data.columns
data.shape

#performing EDA
data.head(10)
data.info()
#its showing no null values

print('Number of books: ', data['ISBN'].nunique())
#270151 number of books
print('Number of users: ',data['User-ID'].nunique())
#92106 number of users
print('Missing data [%]')
round(data.isnull().sum() / len(data) * 100, 4)
#we found age has a round of 26.9446 missing values 
#ploting graph for age 
sns.distplot(data['Age'].dropna(), kde=False)

#checking for outliers 
print('Number of outliers: ', sum(data['Age'] > 100))             
#there are 2910 outliers

data['Book-Rating'] = data['Book-Rating'].replace(0, None) 
#ploting graph for ratings 
sns.countplot(x='Book-Rating', data=data)
# larger number of books are rated with 8 
#finding out average book rating
print('Average book rating: ', round(data['Book-Rating'].mean(), 2))
#its 7.55

#Feature engoineering 
#changing data type 
# Cast to numeric
data['Year-Of-Publication'] = pd.to_numeric(data['Year-Of-Publication'], 'coerse').fillna(2099, downcast = 'infer')
data['Book-Rating'] = data['Book-Rating'].replace(0, None)

#handling outliers 
data['Age'] = np.where(data['Age']>90, None, data['Age'])

#imputing null values 
# Categorical feautes
data[['Book-Author', 'Publisher']] = data[['Book-Author', 'Publisher']].fillna('Unknown')
# Check cat features
data[['Book-Author', 'Publisher']].isnull().sum()

# Age
median = data["Age"].median()
std = data["Age"].std()
is_null = data["Age"].isnull().sum()
rand_age = np.random.randint(median - std, median + std, size = is_null)
age_slice = data["Age"].copy()
age_slice[pd.isnull(age_slice)] = rand_age
data["Age"] = age_slice
data["Age"] = data["Age"].astype(int)
# Checking  Age
data['Age'].isnull().sum()
#succesfully removed null values

#extracting feature
data['Country'] = data['Location'].apply(lambda row: str(row).split(',')[-1])
# Droping irelevant feature
data = data.drop('Location', axis=1)
#preparing dataset

df = data
# Relevant score
df = df[df['Book-Rating'] >= 6]

# Check
df.groupby('ISBN')['User-ID'].count().describe()

df = df.groupby('ISBN').filter(lambda x: len(x) >= 5)
df.groupby('User-ID')['ISBN'].count().describe()

df = df.groupby('User-ID').filter(lambda x: len(x) >= 5)
df.shape

#Building Recommendation system 

df_p = df.pivot_table(index='ISBN', columns='User-ID', values='Book-Rating')
#Select users who liked Lord Of The Rings
lotr = df_p.ix['0345339703'] # Lord of the Rings Part 1
like_lotr = lotr[lotr == 10].to_frame().reset_index()
users = like_lotr['User-ID'].to_frame()

# Trim original dataset
liked = pd.merge(users, df, on='User-ID', how='inner')

rating_count = liked.groupby('ISBN')['Book-Rating'].count().to_frame()

rating_mean = liked.groupby('ISBN')['Book-Rating'].mean().to_frame()

rating_count.rename(columns={'Book-Rating':'Rating-Count'}, inplace=True)

rating_mean.rename(columns={'Book-Rating':'Rating-Mean'}, inplace=True)

liked = pd.merge(liked, rating_count, on='ISBN', how='inner')

liked = pd.merge(liked, rating_mean, on='ISBN', how='inner')

liked['Rating-Mean'] = liked['Rating-Mean'].round(2)
 
liked['Rating-Count'].hist()

C = liked['Rating-Mean'].mean()
C
m = rating_count.quantile(.995)[0] # .9
m

def weighted_rating(x, m=m, C=C):
    v = x['Rating-Count']
    R = x['Rating-Mean']

    return (v/(v+m) * R) + (m/(m+v) * C)
# Create relevant sub-dataset
liked_q = liked.copy().loc[liked['Rating-Count'] >= m]
liked_q.shape

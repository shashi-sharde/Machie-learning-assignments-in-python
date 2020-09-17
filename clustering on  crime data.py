# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 02:28:37 2020

@author: shashi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.cluster import KMeans
import seaborn as sn

crime=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on clustering\\crime_data.csv")
crime


crime.drop(crime.columns[[0]],axis=1,inplace=True)

from sklearn.cluster import KMeans
wcss=[]
plt.figure(figsize=(10, 8))

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(crime)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# by the graph we can decide that the clusters can be 4

my_cluster=KMeans(4)
model=my_cluster.fit(crime)

cluster_labels=pd.Series(model.labels_)
cluster_labels
crime['CLuster']=cluster_labels
crime=crime.iloc[:,]
crime

print(crime.iloc[:,1:].groupby(crime.CLuster).mean())
              #Assault   UrbanPop       Rape  CLuster
#CLuster                                           
#0        112.400000  65.600000  17.270000        0
#1        272.562500  68.312500  28.375000        1
#2        173.285714  70.642857  22.842857        2
#3         62.700000  53.900000  11.510000        3
 #this is the interpretationn of cluster with respect to mean of other variables.
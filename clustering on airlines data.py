# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:17:31 2020

@author: shashi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
air=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on clustering\\EastWestAirlines.csv")
air
pd.set_option('display.max_columns',None)## to see all the columns.
air


air.drop(air.columns[[0]],axis=1,inplace=True)
air

air.isnull().sum()

from sklearn import *
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

air_norm=norm_func(air)
air_norm

# HIERARICHAL CLUSTERING.

from scipy.cluster.hierarchy import *
import scipy.cluster.hierarchy as sci

h=linkage(air_norm,method='complete',metric='euclidean')
plt.figure(figsize=(10,5));plt.title("dendogram");plt.xlabel("index");plt.ylabel("distance")
sci.dendrogram(
        
        h,
        
    
    truncate_mode='lastp',  # show only the last p merged clusters
    p=5,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
        )
plt.show()



from	sklearn.cluster	import	AgglomerativeClustering
 
h_single	=	AgglomerativeClustering(n_clusters=5,	linkage='complete',affinity = "euclidean").fit(air_norm) 

cluster_labels=pd.Series(h_single.labels_)
cluster_labels
air['CLuster']=cluster_labels
air=air.iloc[:,]
air
pd.set_option('display.max_rows',None)
air
air.head(500)

print(air.iloc[:,1:].groupby(air.CLuster).mean())
          
#CLuster   Qual_miles                                                           
#0         88.883768      
#1        208.673846     
#2        248.550699      
#3        347.000000      
#4         32.258065 
# cluster wise average value of qual miles.

# this shows the third cluster is having the highest  qual_miles value.


# K_MEANS CLUSTERING.
air=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on clustering\\EastWestAirlines.csv")
air
pd.set_option('display.max_columns',None)## to see all the columns.
air


air.drop(air.columns[[0]],axis=1,inplace=True)
air

air.isnull().sum()

from sklearn import *
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

air_norm=norm_func(air)
air_norm
from sklearn.cluster import KMeans
wcss=[]
# find the appropriate cluster number
plt.figure(figsize=(10, 8))

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(air_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# by this we can decide that the curve after cluster 5 getting straigh so we can take the cluster no as 5.


my_clust=KMeans(5)
model=my_clust.fit(air_norm)
cluster=model.labels_
cluster
cluster

center_value=model.cluster_centers_
center_value

cluster_labels=pd.Series(model.labels_)
cluster_labels
air['CLuster']=cluster_labels
air=air.iloc[:,]
air
pd.set_option('display.max_rows',None)
air
air.head(500)

plt.figure(figsize=(5, 5))
plt.scatter(air_norm['Balance'], air_norm['Qual_miles'], c=cluster)  
plt.show()


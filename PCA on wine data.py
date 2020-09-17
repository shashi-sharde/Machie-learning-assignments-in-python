# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:54:50 2020

@author: shashi
"""


import pandas as pd 
import numpy as np
wine = pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on pca\\wine.csv")
wine.describe()
wine.head()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

#normalising the numeric data
wine_normal = scale(wine)

pca = PCA(n_components = 6)
pca_values = pca.fit_transform(wine_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="blue")

# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
plt.scatter(x,y,color=["blue"])
# performing clustering 

new_df = pd.DataFrame(pca_values[:,0:4])
# performing k_means 


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
kmeans.labels_
# h clustering 
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(new_df) 
cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
# we got 3 clusters for both kmeans and hclusters
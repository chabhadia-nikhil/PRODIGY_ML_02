# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:50:04 2024

@author: chabh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Retrieving Data
data = "D:\\internships\\prodigy\\Mall_Customers.csv"
data = pd.read_csv(data)

#If spend score is the percentage of their spending then actucal value that is spent is Spend Score*annual income/100
data['Spending (k$)'] = (data['Annual Income (k$)']*data['Spending Score (1-100)'])/100

#Initialising plots.
fig,axis = plt.subplots(1,2,figsize=(15,10))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

#Training
from sklearn.cluster import KMeans
# calculating inertia
inertias =[]
x = np.arange(1,10)
for i in range(1,10):
  kmeans = KMeans(n_clusters = i,n_init=10)
  kmeans.fit(data[['Annual Income (k$)','Spending Score (1-100)']])
  inertias.append(kmeans.inertia_)
axis[1].plot(x,inertias)
axis[1].set_xlabel("No. Clusters")
axis[1].set_ylabel("Inertias")
axis[1].set_title("Elbow graph")
axis[1].set_xlim([0,10])
axis[1].set_ylim([0,1000000])


# From the elbow graph we see that cluster should be 5 as it has low inertia and low number of cluster ( we can say that using elbow method 5 is optimal)
kmeans = KMeans(n_clusters = 5)
predicted_class=kmeans.fit_predict(data[['Annual Income (k$)','Spending Score (1-100)']])
data['Predicted Class'] = predicted_class

#To get location of centroids
cluster_centers=kmeans.cluster_centers_

#To make new dataframe which contains elements with the same predicted class
df0 = data[data['Predicted Class']==0]
df1 = data[data['Predicted Class']==1]
df2 = data[data['Predicted Class']==2]
df3 = data[data['Predicted Class']==3]
df4 = data[data['Predicted Class']==4]

#plotting graph
axis[0].scatter(x=df0['Annual Income (k$)'],y=df0['Spending Score (1-100)'],c="green",marker="*")
axis[0].scatter(x=df1['Annual Income (k$)'],y=df1['Spending Score (1-100)'],c="cyan",marker="*")
axis[0].scatter(x=df2['Annual Income (k$)'],y=df2['Spending Score (1-100)'],c="yellow",marker="*")
axis[0].scatter(x=df3['Annual Income (k$)'],y=df3['Spending Score (1-100)'],c="purple",marker="*")
axis[0].scatter(x=df4['Annual Income (k$)'],y=df4['Spending Score (1-100)'],c="sienna",marker="*")

axis[0].scatter(x=cluster_centers[:,0],y=cluster_centers[:,1],c="red",linewidths=2,edgecolors="black")
axis[0].set_xlabel("Annual Income (k$)")
axis[0].set_ylabel("Spending Score (1-100)")
axis[0].set_title("Annual Income (k$) vs Spending Score (1-100) ")

print(" ")
print("class 0: green color")
print("class 1: cyan color")
print("class 2: yellow color")
print("class 3: purple color")
print("class 4: sienna color")
print(" ")
print("The Red dots are the centroids of that particular class")
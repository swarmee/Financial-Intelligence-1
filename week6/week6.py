# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:09:27 2017

@author: Hou Lan
@time: week 5
@theme: cluster
@content:
E-distance 
k-means/k-mandroid
feature normalization
strength and weaknesses of K-Means
(supplant:DB scan)
CASE STUDY:CLUSTERING TOWN

hint: 
    s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20

k? 找SSE除点数 波谷
画图 elbow点

"""

from sklearn.datasets import make_blobs#做点状图
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np



X,y = make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.5,shuffle=True)
#print(type(X))


fig = plt.figure(figsize=(20,5))
#plt.subplot(1,3,1)
#plt.scatter(X[:,0],X[:,1],c='red',marker='o',s=10)
#model = KMeans(n_clusters=3,init='random',n_init=10,max_iter=300)#依据：平均值的和更小 -> 判断其更集中
#y_km = model.fit_predict(X)#对某一个点有一个预测类值
#plt.subplot(1,3,2)
#plt.scatter(X[y_km==0,0],X[y_km==0,1],s=20,c='lightgreen',marker='s',label='cluster 1')
#plt.scatter(X[y_km==1,0],X[y_km==1,1],s=20,c='orange',marker='o',label='cluster 2')
#plt.scatter(X[y_km==2,0],X[y_km==2,1],s=20,c='lightblue',marker='v',label='cluster 3')
#plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=20,c='red',marker='*')
#plt.legend()
#plt.subplot(1,3,3)

#plt.show()
##
distortions = [] 
for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit(X)
    distortions.append(km.inertia_)
    
plt.subplot(1,3,3)
plt.plot(range(1,11),distortions,marker='o')

#******************************************************************************
#from sklearn.datasets import make_moons

#X,y = make_moons(n_samples=200,noise=0.05)#y是已知为哪一类
#plt.scatter(X[:,0],X[:,1])
#plt.show()
#
#fig = plt.figure(figsize=(20,5))
#plt.subplot(1,2,1)
#km = KMeans(n_clusters=2)
#y_km = km.fit_predict(X)
#plt.scatter(X[y_km==0,0],X[y_km==0,1],c='red',marker='o',s=20)
#plt.scatter(X[y_km==1,0],X[y_km==1,1],c='green',marker='*',s=20)
#plt.xlabel('x')
#plt.xlabel('y')
#plt.title('KMeans')



#******************************************************************************
from sklearn.cluster import DBSCAN
plt.subplot(1,2,1)
km = DBSCAN(eps=0.2,min_samples=2,metric='euclidean')
DBSCAN()
y_km = km.fit_predict(X)

plt.scatter(X[y_km==0,0],X[y_km==0,1],c='red',marker='o',s=20)
plt.scatter(X[y_km==1,0],X[y_km==1,1],c='green',marker='*',s=20)
plt.title('DBSCAN')
plt.show()


















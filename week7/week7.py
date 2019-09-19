# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:06:53 2017

@author: Hou LAN

样本点：

直接密度可达、密度可达(不对称)、密度相连（对称）。

算法：深度遍历

怎么选？
eps每两点举例图，选拐点。
minpts
二维数据时取4

优点：环形带状
缺点：集中密度差异大的数据
    度量距离不同 会有很大差别

复杂度:时间复杂度:n*log(n)
    空间复杂度：n


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics



#train = pd.read_csv('optdigits.tra',header=None)
#test = pd.read_csv('optdigits.tes',header=None)
#
#X_train = train[np.arange(64)]
#Y_train = train[64]
#
#X_test = train[np.arange(64)]
#Y_test = train[64]
#
#km = KMeans(n_clusters=10)
#km.fit(X_train)
#y_pred = km.predict(X_test)
#score = metrics.adjusted_rand_score(Y_test,y_pred)
#print(score)

"""
silhouette coefficient

cohesion
separation

sc = bi-ai/max(ai,bi)
"""

x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])

alist = list(zip(x1,x2))
data = np.array(alist)
X = pd.DataFrame(data)

fig = plt.figure()
plt.subplot(3,2,1)
plt.scatter(x1,x2)
plt.show()

colors = ['b','g','r','c','y','m','k','b']
markers = ['o','s','D','v','^','p','*','+']
clusters = [2,3,4,5,8]
subplot_counter = 1
sc_scores = []

for t in clusters:
    subplot_counter +=1
    plt.subplot(3,2,subplot_counter)
    km = KMeans(n_clusters=t)
    km.fit(data)
#    print(km.labels_)

    for i,l in enumerate(km.labels_):
        plt.plot(x1[i],x2[i],color=colors[l],marker=markers[l])
    sc_score = metrics.silhouette_score(X,km.labels_,metric="euclidean")
    sc_scores.append(sc_score)

    plt.title('k=%s,SC=%0.03f'%(t,sc_score),fontsize='small')
    
plt.show()








#采集数据 评价 贡献 案例论证
#神经网络预测


























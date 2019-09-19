# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:36:56 2017

@author: Administrator
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn import preprocessing
import pickle 
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import pandas as pd

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import ward,dendrogram,fcluster

#打开文件
f=open('stocks.txt','rb')

try:
    all_that_data = pickle.load(f)
except IOError as error:
    print(error)

price=pd.DataFrame({tic:data['Adj Close'] for tic,data in all_that_data.items()})

price = price.drop(price.index[1])
price = price.drop(price.index[11])
returns= price.pct_change()
returns = returns.drop(returns.index[0])
treturns = returns.T
nana=[]
for i in range(len(treturns.index)):
    for j in range(len(treturns.columns)):
        if math.isnan(treturns.ix[i,j]):
            nana.append(i)
            break
matData = treturns.as_matrix()
ll = list(set(list(range(7106))).difference(set(nana)))
matData = matData[ll]                    
###########################################

#预处理
matData = preprocessing.scale(matData)
###########################################

#K-means,看结果
model = KMeans(n_clusters=3,init='random',n_init=10,max_iter=300)
y_km = model.fit_predict(matData)
II = treturns.index[ll]
c0=[]
c1=[]
c2=[]
count=0
for i in y_km:
    if i == 0:
        c0.append(II[count])
    if i == 1:
        c1.append(II[count])
    if i == 2:
        c2.append(II[count])
    count = count+1

#看曲线
#distortions = []
#for i in range(1,13):
#    km = KMeans(n_clusters=i)
#    km.fit(matData)
#    distortions.append(km.inertia_)
#plt.plot(range(1,13),distortions)

#############################################

#层次聚类
#fig = plt.figure()
#for j in  range(5,9):
#    ward = AgglomerativeClustering(n_clusters=j, linkage='ward').fit(matData)
#    label = ward.labels_                
#    clus = set(label)
#    county = {} 
#    for i in clus:
#        county[i] = list(label).count(i)
#    plt.subplot(1,4,j-4)
#    plt.bar(county.keys(),county.values())        

#牛一点的层次聚类
#distance = pdist(matData)
#linkresult = ward(distance)
#DL = dendrogram(linkresult)

#fclu = fcluster(linkresult,t=i,criterion='maxclust')

#################################################

#   LAG
def buildLaggedFeatures(s,lag=2,dropna=True):
    '''
    Builds a new DataFrame to facilitate regressing over all possible lagged features
    '''
    if type(s) is pd.DataFrame:
        new_dict={}
        for col_name in s:
            new_dict[col_name]=s[col_name]
            # create lagged Series
            for l in range(1,lag+1):
                new_dict['%s_lag%d' %(col_name,l)]=s[col_name].shift(l)
        res=pd.DataFrame(new_dict,index=s.index)
    
    elif type(s) is pd.Series:
        the_range=range(lag+1)
        res=pd.concat([s.shift(i) for i in the_range],axis=1)
        res.columns=['lag_%d' %i for i in the_range]
    else:
        print('Only works for DataFrame or Series')
        return None
    if dropna:
        return res.dropna()
    else:
        return res

corrma = []
for c in [c0,c1,c2]:
    ReturnsLag = buildLaggedFeatures(returns[c],lag=1,dropna=False)
    lori = []
    lola = []
    l = list(ReturnsLag.columns)
    count=0
    for i in l:
        if '_' not in i:
            lori.append(count)
        else:
            lola.append(count)
        count = count+1
           
    CorrTotall = ReturnsLag.corr()
    CorrLast = CorrTotall.ix[lori,lola]
    CorrLast = pd.DataFrame.abs(CorrLast)
    corrarray = CorrLast.T.mean()
    corrma.append(corrarray)
    print(corrarray.idxmax())
      
    
    
    

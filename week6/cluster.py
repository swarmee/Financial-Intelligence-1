# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:07:45 2017

@author: Hou LAN
"""


import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import time


df = pd.read_csv('fin_2017-04-09_2391.csv',index_col=[0])
#lst_tic = list(df.tic.unique())
#df.index = df.Date
#df.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Volume','datadate','Adj Close','prccd_norm'],axis=1,inplace=True)
#
#df = df[df.tic!='UA']
#df = df[df.tic!='GPIA']
#df = df[df.tic!='MSG']
#df.drop('2015-08-03',axis=0,inplace=True)
#result = pd.DataFrame()
#for t in lst_tic:
#    df_tmp = df[df.tic==t]
#    df_tmp['ret'] = df_tmp['Adj Close'].pct_change()
#    result = pd.concat([result,df_tmp])
#
#result = {}
#for t in lst_tic:
#    result[t] = df.ix[df.tic==t,'ret']
##    
#
#result  =pd.DataFrame.from_dict(result,orient='columns')
#df = pd.read_csv('fin_2017-04-14_2391.csv',index_col=[0])
#df.drop('2015-08-03',axis=0,inplace=True)
#sample = df[:20]

#******************************preprocessing***********************************

for i in range(2391):
    if sample.ix[:,i].isnull().any():
        print(i)
#sample.drop('UA',axis=1,inplace=True)
#sample.drop('GPIA',axis=1,inplace=True)
#sample.drop('MSG',axis=1,inplace=True)
#data = sample.T
#dataMatrix = data.as_matrix()
#dataMatrix = preprocessing.scale(dataMatrix)

#normalized = lambda x : (x-x.mean())/x.std()
#result = pd.DataFrame()
#for i in lst_tic:
#    df_tmp = df[df.tic==i]
#    df_tmp['prccd_norm'] = normalized(np.array(df_tmp['Adj Close']))
#    result = pd.concat([result,df_tmp])
#sample = result[:20]


#***********,*******************clustering**************************************

#dict_df = {}
#lst_tic_part = list(result_prt.tic.unique())
#for i in lst_tic_part:
#    dict_df[i] = result_prt.ix[result_prt.tic==i,'prccd_norm']
#        
#result_prt_df =pd.DataFrame.from_dict(dict_df,orient='columns')
#sample = result_prt_df.ix[:,:100]
#
#numpyMatrix = sample.T.as_matrix()

#
#distortions = [] 
#sc_scores = []
#for i in range(2,20):
#    km = KMeans(n_clusters=i,init='k-means++')
#    km.fit(dataMatrix)
#    distortions.append(km.inertia_)
#    sc_score = metrics.silhouette_score(dataMatrix,km.labels_,metric="euclidean")
#    sc_scores.append(sc_score)
##  preprocessing.scale(sp[sp.tic=='A'].ret)
#plt.subplot(1,1,1)
#plt.plot(range(2,20),distortions,marker='o')
#names = np.array(df.columns)
#
#
###    
#
#dataMatrix = df.T.as_matrix()
#model = KMeans(n_clusters=3,init='k-means++')#依据：平均值的和更小 -> 判断其更集中
#y_km = model.fit_predict(dataMatrix)#对某一个点有一个预测类值


#ret_lag_1 = pd.DataFrame()
#ret_lag_2 = pd.DataFrame()
#ret_lag_3 = pd.DataFrame()
#cluster_1 = df[names[y_km==0]]
#cluster_2 = df[names[y_km==1]]
#cluster_3 = df[names[y_km==2]]
###tic_2388 = list(sample.columns)
#tic_1 = list(cluster_1.columns)
#tic_2 = list(cluster_2.columns)
#tic_3 = list(cluster_3.columns)
#for i in tic_1:
#    ret_lag_1[i+'_lag'] = cluster_1[i].rolling(window = 5, center = False).mean()
#for i in tic_2:
#    ret_lag_2[i+'_lag'] = cluster_2[i].rolling(window = 5, center = False).mean()
#for i in tic_3:
#    ret_lag_3[i+'_lag'] = cluster_3[i].rolling(window = 5, center = False).mean()
#df_1 = ret_lag_1.join(cluster_1)
#df_2 = ret_lag_2.join(cluster_2)   
#df_3 = ret_lag_3.join(cluster_3)       
#df_1.fillna(0,inplace=True)
#df_2.fillna(0,inplace=True)
#df_3.fillna(0,inplace=True)
#
#
#
#
#corr_df_1 = df_1.corr(method='pearson')
#corr_df_2 = df_2.corr(method='pearson')
#corr_df_3 = df_3.corr(method='pearson')
#leader_1 = np.mean(abs(corr_df_1.ix[len(cluster_1.columns):,:(len(cluster_1.columns)-1)]).T).idxmax()
#leader_2 = np.mean(abs(corr_df_2.ix[len(cluster_2.columns):,:(len(cluster_2.columns)-1)]).T).idxmax()
#leader_3 = np.mean(abs(corr_df_3.ix[len(cluster_3.columns):,:(len(cluster_3.columns)-1)]).T).idxmax()
##plt.figure()
#with pd.plot_params.use('x_compat', True):
#    ax = cluster_2.plot(style='--',legend=False,title='cluster 2')
#    ax = cluster_2[leader_2].plot(color='r',linewidth='3',legend=False)
#ax.set_ylabel('ret')
#ax.set_xticks(cluster_3.index)
#ax.xaxis.set_major_locator(MaxNLocator(nbins=30, integer=True))
#    



###############################################################################
#sp = df[df.datadate<20150901]
#ret = sp[["ret", "tic"]].set_index("tic", append=True).ret.unstack("tic")
#dataMatrix = ret.T.as_matrix()
#dataMatrix = preprocessing.scale(dataMatrix.T).T
#df_ret = pd.DataFrame(dataMatrix.T,columns=ret.columns,index=ret.index)  
#distortions = [] 
#sc_scores = []
#for i in range(1,20):
#    km = KMeans(n_clusters=i)
#    km.fit(dataMatrix)
#    distortions.append(km.inertia_)
#    sc_score = metrics.silhouette_score(dataMatrix,km.labels_,metric="euclidean")
#    sc_scores.append(sc_score)
#plt.subplot(1,1,1)
#plt.plot(range(1,20),distortions,marker='o')
#names = np.array(df_ret.columns)                              
#when you use this function to reshape your dataframe, make sure no null value 
#in your columns
#lst_tic = list(sp.tic.unique())
#df_tmp = pd.DataFrame()
#for tic in lst_tic:
#    
#    
#test = sp[["ret", "tic"]].set_index("tic", append=True).ret.unstack("tic")



###############################################################################




    
    
def cal_ret(df_stock,lst_tic):
    '''accept a dataframe of stock price collection with firm tickers;
              a list of firm tickers.
       return the dataframe with calculated returns.'''

    stock_ret = pd.DataFrame()
    for t in lst_tic:
            df_tmp = df_stock[df_stock.tic==t]
            df_tmp['ret'] = df_tmp['Adj Close'].pct_change()
            stock_ret = pd.concat([stock_ret,df_tmp])
    return stock_ret






#get sample of data structrue
#from pandas.tools.plotting import table
#
#ax = plt.subplot(111,frame_on=False) # no visible frame
#ax.xaxis.set_visible(False)  # hide the x axis
#ax.yaxis.set_visible(False)  # hide the y axis
#
#table(ax, stock_df.head(),loc='center')  # where df is your data frame
#plt.savefig('sample.png')



#*********************************************#
#
#    Hierarchical Clustering
#
#*********************************************#


#from scipy import cluster
#
#Z = cluster.hierarchy.linkage(dataMatrix,'ward')
##cluster.hierarchy.dendrogram(Z)
#score = metrics.silhouette_score(dataMatrix,cut,metric="euclidean")




#cluster_output = pandas.DataFrame({'team':df.teamID.tolist() , 'cluster':assignments})




###############################################################################




#*********************************************#
#
#    SVM predicting
#
#*********************************************#

from sklearn.svm import SVR



def predict_prices(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))
    svr_lin = SVR(kernel='linear',C=1e3, )
    svr_poly = SVR(kernel='poly',C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf',C=1e3, gamma=0.1)
    svr_lin.fit(dates,prices)
    svr_poly.fit(dates,prices)
    svr_rbf.fit(dates,prices)
    
    plt.scatter(dates,prices,color='black',label='Data')
    plt.plot(dates,svr_rbf.predict(dates),color='red',label='RBF model')
    plt.plot(dates,svr_lin.predict(dates,color='green',label='Linear model'))
    plt.plot(dates,svr_poly.predict(dates,color='blue',label='Polynomial model'))
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0],svr_lin.predict(x)[0],svr_poly.predict(x)[0]













##predictiton though deep learning
#
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense,Activation,Dropout
#
##buid model
model = Sequential()
model.add(LSTM(input_dim=1,output_dim=50,return_sequences=True))
model.add(LSTM(100,return_sequences=False))
model.add(Dense(output_dim=1))
model.add(Activation('linear'))
start = time.time()
model.compile(loss='mse',optimizer='rmsprop')
print('compilation time:',time.time()-start)
#
##train model
#model.fit(X_train,y_train,batch_size=512,nb_epoch=1,validation_split=0.05)






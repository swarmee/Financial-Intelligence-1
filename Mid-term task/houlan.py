# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:48:22 2017

@author: Hou LAN 41423018
"""
import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.stats.stats import pearsonr
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors




#*********************************************#
#
#    TASK 1
#
#*********************************************#




##read the firm list 

#firm_lst = pd.read_excel('firm list.xlsx',names=['COMN'],header=None)
##
###get remote data
start = datetime.datetime(2015,12,31)
end = datetime.datetime(2016,1,8)
#
#lst_tic = list(firm_lst.COMN)
#stock_df = pd.DataFrame()
#
#for tic in lst_tic:
#    try:
#        f = web.DataReader(tic,'yahoo',start,end)
#        f.reset_index(level=0,inplace=True)
#        f['tic'] = tic
#        stock_df = pd.concat([stock_df,f])
#    except Exception:
#        pass
#stock_df['Date'] = stock_df['Date'].astype(str)
#stock_df.index = stock_df.Date #set Date AS INDEX for convenience

#calculate price changes 
#stock_ret = pd.DataFrame()
#for t in lst_tic:
#    df_tmp = stock_df[stock_df.tic==t]
#    df_tmp['ret'] = df_tmp['Adj Close'].pct_change()
#    stock_ret = pd.concat([stock_ret,df_tmp])

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
#    TASK 2
#
#*********************************************#


#preprocessing
#stock_ret.index = stock_ret.Date
#nullret_date = stock_ret.ix[stock_ret.ret.isnull(),'Date'].unique()
#stock_ret.drop(nullret_date,axis=0,inplace=True)

#get normalized price
#price_df = stock_df[["Adj Close", "tic"]].set_index("tic", append=True)['Adj Close'].unstack("tic")
#price_matrix = price_df.T.as_matrix()
#price_matrix = preprocessing.scale(price_matrix.T).T
#norm_price = pd.DataFrame(price_matrix.T,columns=price_df.columns,index=price_df.index)  


#get normalized return                                 
#ret_df = stock_ret[["ret", "tic"]].set_index("tic", append=True).ret.unstack("tic")
#ret_matrix = ret_df.T.as_matrix()
#ret_matrix = preprocessing.scale(ret_matrix.T).T
#norm_ret = pd.DataFrame(ret_matrix.T,columns=ret_df.columns,index=ret_df.index)  


#visulization
#fig = plt.figure()
#ax1 = norm_price.ix[:,:5].plot(grid=True,legend='Best',title='TOP 5 Stock Price From 2015-12-31 To 2016-01-08')
#ax1.set_ylabel('Normalized Price')
#plt.savefig('normalized_price.png')

#ax2 = ret.ix[:,:5].plot(grid=True,legend='Best',title='TOP 5 Stock Return From 2016-01-01 To 2016-01-08')
#ax2.set_ylabel('Normalized Return')   
#plt.savefig('normalized_ret.png')    


#*********************************************#
#
#    TASK 3
#
#*********************************************#

#calculate the related matrix and find the most related two firms 
#corr_ret = norm_ret.corr(method='pearson')
#corr_ret[corr_ret==1] = 0 
#conm = corr_ret.columns
#idy, idx = np.where(corr_ret == corr_ret.max().max())
#conm_1,conm_2 = conm[idx] # find most related two stocks

#*********************************************#
#
#    TASK 4
#
#*********************************************#


#K-means

#distortions_k = [] 
#sc_scores_k = []
#for i in range(2,30):
#    km = KMeans(n_clusters=i)
#    km.fit(ret_matrix)
#    distortions_k.append(km.inertia_)
#    sc_score = metrics.silhouette_score(ret_matrix,km.labels_,metric="euclidean")
#    sc_scores_k.append(sc_score)
#
#ax3 = plt.subplot(1,1,1)
#ax3.set_title('find elbow point in K-means')
#ax3 = plt.plot(range(2,30),distortions_k,marker='o',c='blue')
#plt.savefig('kmeans.png')
#model = KMeans(n_clusters=4)#依据：平均值的和更小 -> 判断其更集中
#y_km = model.fit_predict(ret_matrix)#对某一个点有一个预测类值




#DBSCAN

#km = DBSCAN(eps=0.8, min_samples=4, metric='euclidean')
#km.fit(ret_matrix)
#sc_score_d = metrics.silhouette_score(ret_matrix,km.labels_,metric="euclidean")

#nbrs = NearestNeighbors(n_neighbors=len(ret_matrix)).fit(ret_matrix)
#distances, indices = nbrs.kneighbors(ret_matrix)
#ax4 = plt.subplot(1,1,1)
#ax4.set_title('find optimal eps in DBSCAN')
#ax4 = plt.plot(sorted(distances[-1]))
#plt.savefig('dbscan.png')


#*********************************************#
#
#    OPEN TASK
#
#*********************************************#

                 
sp_500 = web.get_data_yahoo('^GSPC',start,end)['Adj Close'] 
nasdaq = web.get_data_yahoo('^IXIC',start,end)['Adj Close'] 
dow = web.get_data_yahoo('^DJI',start,end)['Adj Close'] 
total = pd.DataFrame([sp_500,nasdaq,dow]).T
total.columns = ['sp_500','nasdaq','dow']
fig = plt.Figure()
ax = total.plot(grid=True,legend='best')
ax.set_title('S&P·NASDAQ·DOW INDEX From 2015-12-31 To 2016-01-08')

 


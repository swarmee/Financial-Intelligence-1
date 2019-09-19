# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:33:31 2017

@author: Hou LAN
"""
#first make sure packages are loaded
import numpy as np
import pandas as pd
import time

from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans

from scipy import cluster
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib.externals import six
from matplotlib.ticker import FuncFormatter, MaxNLocator





#*********************************************#
#
#    Data Preprocessing
#
#*********************************************#
    


#check stock data
def check_colinfo(df_stock,col_name):
    '''accept a dataframe of stock price collection with firm tickers and timestamp;
              a list of firm tickers.
       print the anomalous column value information.
    '''
    #if there is null value in column
    if df_stock[col_name].isnull().any():
        #get the array of the index of null value
        idx = np.where(df_stock[col_name].isnull())
        #get the tickers of firms with null value
        null_tic = list(df_stock.ix[idx[0]].tic.unique()) 
        if len(null_tic) <=10:
            for i in null_tic:
                print(col_name,'of',i,'have/has NULL value, please CHECK!')
        else:
            print(len(null_tic),'stocks have NULL value, please CHECK!')
            
    #if there is negative value
    if (df_stock[col_name]<0).any():
        neg_tic = list(df_stock.ix[df_stock[col_name]<0,'tic'].unique())
        if len(neg_tic) <=10:
            for i in neg_tic:
                print(col_name,'of',i,'have/has negative value(s), please CHECK!')
        else:
            print(len(neg_tic),'stocks have negative value(s), please CHECK!')
        
    #if there is repetitive record on the same day
    #for convenience we assume date as one seperate float column
    if df_stock.duplicated(['tic','datadate']).any():
        dup_tic = list(df_stock.ix[np.where(df_stock.duplicated(['tic','datadate']))[0],'tic'].unique())
        if len(dup_tic) <=10:
            for i in dup_tic:
                print(col_name,'of',i,'have/has repetitive value(s), please CHECK!')
        else:
            print(len(dup_tic),'stocks have repetitive value(s), please CHECK!')
  

    
# plot sample of dataframe    
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    '''accept a dataframe
       return an ax to see dataframe clearly
    '''
    #A5A5A5
    #change decimal if necessary
    data = np.round(data, decimals=5)
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

#
##*********************************************#
##
##    Cluster
##
##*********************************************#
#
#
#                                  
##K-means
#
##find optimal K by finding elbow point
#def plot_k(data_matrix,range_max=15):
#    '''accept a data matrix and a right end 
#       return a plot to visualize optimal k in K-Means 
#    '''
#    distortions_k = []
#    for i in range(1,range_max):
#        km = KMeans(n_clusters=i)
#        km.fit(data_matrix)
#        distortions_k.append(km.inertia_)
#
#    ax = plt.subplot(1,1,1)
#    ax.set_title('find elbow point in K-means')
#    ax = plt.plot(range(1,range_max),distortions_k,marker='o',c='#40466e')
#    
#    return ax
#
#
#
##validation by silhouette_score
#
#def validate_km(data_matrix,range_max=15):
#    '''accept a data matrix and a right end 
#       return silhouette score of K-Means
#    '''
#    sc_scores_k = []
#    for i in range(2,range_max):
#        km = KMeans(n_clusters=i)
#        km.fit(data_matrix)
#        sc_score = metrics.silhouette_score(data_matrix,km.labels_,metric="euclidean")
#        sc_scores_k.append(sc_score)
#        
#    return sc_scores_k
#    
#
#
#
#def validate_fcluster(data_matrix,range_max=15):
#    '''accept a data matrix and a right end 
#       return silhouette score of hierarchical cluster
#    '''
#    sc_scores_f = []
#    Z = cluster.hierarchy.linkage(data_matrix,'ward')
#    for i in range(2,range_max):
#        fcluster = cluster.hierarchy.fcluster(Z,i,criterion='maxclust')
#        sc_score = metrics.silhouette_score(data_matrix,fcluster,metric="euclidean")
#        sc_scores_f.append(sc_score)
#        
#    return sc_scores_f
#   
#
#
#
#
##*********************************************#
##
##    Correlation 
##
##*********************************************#
#
#
#
#def find_leadstock(cluster):
#    
#    tic = list(cluster.columns.unique())
#    #calculate simple moving average to find "group leader" with maximum correlation with "future values" of other stocks
#    ret_lag = pd.DataFrame()
#    for i in tic:
#        ret_lag[i+'_lag'] = cluster[i].rolling(window = 5, center = False).mean()
#    
#    df_tmp = ret_lag.join(cluster)
#    df_tmp.fillna(0,inplace=True)
#    corr_df = df_tmp.corr(method='pearson')
#    leader = np.mean(abs(corr_df.ix[len(cluster.columns):,:(len(cluster.columns)-1)]).T).idxmax()
#    
#    return leader
#    
#
#
#
#def plot_cluster(cluster,leader,titlename):
#    with pd.plot_params.use('x_compat', True):
#        ax = cluster.plot(style='--',legend=False,title=titlename)
#        ax = cluster[leader].plot(color='r',linewidth='3',legend=True)
#    ax.set_ylabel('Normalized Return')
#    return ax
#
#
#
#
#
##*********************************************#
##
##    Regression
##
##*********************************************#
#
#
#
##SVM predicting
#
#
#from sklearn.svm import SVR
#from sklearn import cross_validation
#
#
#
#def svr_predict_price(dates, prices, x):
#    date_tick = dates #use as tick
#    dates = np.reshape(list(range(len(dates))),(len(dates), 1)) # converting to matrix of n X 1
#    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
#    svr_lin = SVR(kernel= 'linear', C= 1e3)
#    svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
#    svr_rbf.fit(dates, prices) # fitting the data points in the models
#    svr_lin.fit(dates, prices)
#    svr_poly.fit(dates, prices)
#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1)
#    ax.set_xlim(0,len(date_tick))
#
#    ax.scatter(np.reshape(list(range(len(date_tick))),(len(date_tick), 1)), prices, color= 'black', label= 'Data') # plotting the initial datapoints 
#    ax.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by rbf kernel
#    ax.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
#    ax.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
#    ax.set_xlabel('Date')
#    ax.set_ylabel('Price')
#    ax.set_title('Support Vector Regression')
#    ax.xaxis.set_major_locator(MaxNLocator(nbins=len(date_tick), integer=True))
#    ax.set_xticklabels(date_tick)
#    plt.legend(loc='best')
#    plt.xticks(rotation=45)
#    plt.tight_layout()
#    plt.show()
#    
#    loo = cross_validation.LeaveOneOut(len(prices))
#    score_rbf = cross_validation.cross_val_score(svr_rbf, dates, prices, scoring='mean_squared_error', cv=loo,)
#    score_lin = cross_validation.cross_val_score(svr_lin, dates, prices, scoring='mean_squared_error', cv=loo,)
#    score_poly = cross_validation.cross_val_score(svr_poly, dates, prices, scoring='mean_squared_error', cv=loo,)
#
#    return score_rbf, score_lin, score_poly
#
#
#
#
#
#if __name__ == '__main__':
#    #read in stock data csv
#    df = pd.read_csv('fin_2017-04-23_2390.csv')
#    #get stock return
#    df['ret'] = df.groupby('tic')['Adj Close'].pct_change().fillna(0)
#    df.index = df.Date
#    
#    #visualize original stock dataframe from Yahoo Finance API
##    f1 = plt.figure()
##    ax1 = f1.add_subplot(111)
#    ax1 = render_mpl_table(df.head(), header_columns=0, col_width=2.0)
#    ax1.set_title('Sample of Original Stock DataFrame')
#
#    #get return matrix
#    ret_df = df[["ret", "tic"]].set_index("tic", append=True).ret.unstack("tic")
#    ret_matrix = ret_df.T.as_matrix()
#    ret_matrix = preprocessing.scale(ret_matrix.T).T
#    ret_df_norm = pd.DataFrame(ret_matrix.T,index=ret_df.index,columns=ret_df.columns)
#
#    #visualize optimal K in K-means
#    f2 = plt.figure()
#    ax2 = f2.add_subplot(111)
#    ax2.set_title('To find elbow point in K-Means')
#    ax2 = plot_k(ret_matrix)
#
#    sc_scores_k = validate_km(ret_matrix)
#    model = KMeans(n_clusters=2)
#    y_km = model.fit_predict(ret_matrix)
#    
#    Z = cluster.hierarchy.linkage(ret_matrix,'ward')
#    #visualize hierarchy dendrogram
#    f3 = plt.figure()
#    ax3 = f3.add_subplot(111)
#    ax3.set_title('Dendrogram of Hierarchy Cluster')
#    ax3 = cluster.hierarchy.dendrogram(Z)
#    fcluster = cluster.hierarchy.fcluster(Z,2,criterion='maxclust')
#    sc_scores_f = validate_fcluster(ret_matrix)
#    
#    #visualize the comparation between K-Means and hierarchy    
#    #df_km_fclu = pd.DataFrame([sc_scores_k,sc_scores_f],index=['sc_scores_k','sc_scores_f'],columns=list(range(2,15))).T                     
#    #N = 3
#    #ind = np.arange(N)  # the x locations for the groups
#    #width = 0.27       # the width of the bars 
#    
#    #f = plt.figure()
#    #ax = f.add_subplot(111)
#    
#    #rects1 = ax.bar(ind+0.2, sc_scores_k[:3], width, color='#9999ff',alpha=0.6)
#    #rects2 = ax.bar(ind+width+0.2, sc_scores_f[:3], width, color='#ff9999',alpha=0.6)
#    
#    #ax.set_title('Comparison between KMeans and Hierarchy Cluster')
#    #ax.set_ylabel('Silhouette score')
#    #ax.set_xticks(ind+width+0.2)
#    #ax.set_xticklabels( ('Cluster 2', 'Cluster 3', 'Cluster 4') )
#    #ax.legend((rects1[0], rects2[0]), ('KMeans', 'Hierarchy') )
#
#
#    #get two cluster dataframe according to analysing above and use K-Means as a better clustering
#    names = np.array(ret_df.columns)
#    cluster_1 = ret_df_norm[names[y_km==0]]
#    cluster_2 = ret_df_norm[names[y_km==1]]
#
#    
#    leader_1 = find_leadstock(cluster_1)
#    leader_2 = find_leadstock(cluster_2)
#
#
#
#    #visualize cluster
#    plot_cluster(cluster_1,leader_1,'Cluster 1')
##    plot_cluster(cluster_2,leader_2,'Cluster 2')
#
#    plt.show()
#    
#    #SVM prediction
##    df.reset_index(drop=True,inplace=True)
##    dates = df[df.tic==leader_2].Date
##    prices = df.ix[df.tic==leader_2,'Adj Close']
#
#    #take much time
##    x,y,z = svr_predict_price(dates,prices,len(dates))
#
#
#
#
#
#
#
#
#
#































    



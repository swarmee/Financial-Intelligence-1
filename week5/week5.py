# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:08:10 2017

@author: Hou LAN
"""

#*******************pandas************************
import numpy as np
import pandas as pd

#created by dict
df1 = pd.DataFrame({'one':np.random.randn(4),'two':np.linspace(1,4,4),'three':['zhangsan','lisi',999.99,1]})
index = pd.date_range('20161001',periods=4)
df1.set_index(index,inplace=True)

#created by array
a = np.random.randn(4,6)
df2 = pd.DataFrame(a,index=index,columns=list('ABCDEF'))

#created by other df
df3 = df2[['A','B','C']].copy()
df3 = df2[:3] 
df3 = df2['2016-10-01':'2016-10-04']#包括首尾 不像index
df3 = df2.iloc[:,0:5]



df3 = df2['2016-10-01':'2016-10-04']
try:    
    f = open('sample.txt','a+')
    contents = 'Hello,world'
    f.write(contents)
except IOError as err:
    print(str(err))
finally:
    f.close()

with open('sample.txt','a+') as out:
    contents = 'hello,world'
    out.write(contents)

#*******************get remote data************************
import pandas_datareader.data as web
import datetime
import pandas as pd
start  = datetime.datetime(2017,1,1)
end = datetime.datetime.now()

#AAPL = web.DataReader('AAPL','yahoo',start,end)
#print(AAPL.index)
#
#try:
#    AAPL.to_csv('AAPL.csv')
#
#except IOError as err:
#    print(err)
#
#try:
#    data = pd.read_csv('AAPL.csv')
#    print(data.tail(3))
#except IOError as err:
#    print(err)



#*******************plot************************
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pickle

#np.random.seed(1000)
#y = np.random.standard_normal(20)
#x = range(len(y))
#plt.plot(x,y)
#plt.show()
#
#data = AAPL[:20]
#plt.plot(AAPL.index,AAPL['Adj Close'],marker='o',linestyle='--',color='g')
#pylab.rcParams['figure.figsize']=(10,5)
#plt.show()    
#
#
#fig = plt.figure()
#ax_price = fig.add_subplot(1,2,1)
#ax_volume = fig.add_subplot(1,2,2)
#ax_price.plot(data.index,data['Adj Close'])
#ax_volume.plot(data.index,data['Volume'])
#ax_price.set_xticklabels(data.index,rotation=45)
#ax_volume.set_xticklabels(data.index,rotation=45)
#ax_price.legend()
#ax_volume.legend()

#ax_volume.xticks(rotation=45)
#龙头股：先涨一点，别的跟着涨(简单版)；同时涨 考虑时差
all_data = {}
lst_tic = ['AAPL','IBM','GOOG']
for tic in lst_tic:
    try:
        all_data[tic] = web.DataReader(tic,'yahoo',start,end)
        
    except Exception:
        pass
    
    
f = open('stocks.txt','wb')
try:
    pickle.dump(all_data,f)
except IOError as err:
    print(err)
finally:
    f.close()

f = open('stocks.txt','rb')    
try:
    all_data = pickle.load(f)
except IOError as err:
    print(err)

print('hi,buddy,this is all you want \n',all_data)
    
from pandas import DataFrame    



price = DataFrame({tic:data['Adj Close'] for tic,data in all_data.items()})   

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(price.index,price.AAPL,linestyle='--',label='Apple')
ax.legend(loc='best')

plt.show()
#get returns
returns = price.pct_change()

cor = returns.AAPL.corr(returns.IBM)
print(returns.corr().idxmin())

aSeries =  returns.corrwith(returns.GOOG)
aSeries.order()
print('GOOG\n',aSeries.order(ascending=False))

#某一个时段 哪些公司 变动具有很强的相关性 股灾 政策影响的时间段 源代码 word文件画图 一个表里把变化趋势显示出来
#GOOGLE的前五家 set 取交集
#    
    
    

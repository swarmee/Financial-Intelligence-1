# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:56:20 2017

@author: Hou Lan
@41423018
"""
import pandas as pd
import numpy as np



#***********************T1********************************
def cal_changepercentage(df,tic):
    lst_p = list(df.loc[:,tic])
    lst_p.insert(0,np.nan)
    lst_p = lst_p[:-1]
    df['ptmp']=lst_p
    col_name = tic + 'changepercentage'
    df[col_name] = abs((df[tic]-df['ptmp'])/df['ptmp'])
    return df
        
       

    
s1 = [0.8, 0.85, 0.86, 0.84, 0.9, 0.99]
s2 = [0.7, 0.75, 0.77, 0.75, 0.72, 0.77]
time = ['9:00','9:10','9:20','9:30','9:40','9:50']
df = pd.DataFrame()
df['s1'] = s1
df['s2'] = s2
df.index = time

cal_changepercentage(df,'s1')
cal_changepercentage(df,'s2')

print('min percentage of s1:',min(df.s1changepercentage[1:]),'time:',df[df.s1changepercentage==min(df.s1changepercentage[1:])].index)
print('max percentage of s1:',max(df.s1changepercentage[1:]),'time:',df[df.s1changepercentage==max(df.s1changepercentage[1:])].index)
print('min percentage of s2:',min(df.s2changepercentage[1:]),'time:',df[df.s2changepercentage==min(df.s2changepercentage[1:])].index)
print('max percentage of s2:',max(df.s2changepercentage[1:]),'time:',df[df.s2changepercentage==max(df.s2changepercentage[1:])].index)


#***********************T2********************************
def licai(base,rate,days):
    result = base
    times = 365//days
    for i in range(times):
        result = result + result*rate/365*days
    return result
    
#invest 100000000 365 days 
invest_amt = 100000000
rate1 = 0.07
days1 = 14
rate2 = 0.063
days2 = 8
t1 = licai(invest_amt,rate1,days1)
t2 = licai(invest_amt,rate2,days2)

if t1>t2:
    print('choose rate:',rate1,'days:',days1)
elif t1<t2:
    print('choose rate:',rate2,'days:',days2)
elif t1==t2:
    print('two investment are the same')



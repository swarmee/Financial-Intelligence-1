# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:08:06 2017

@author: Hou Lan
"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import os
from datetime import *
import numpy as np


def mplot(a, title):

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(a.index,a,'-',linewidth=2)
    ax1.plot(a.index,a,'yo', markersize=2)
    
    date_format = mpl.dates.DateFormatter("%Y-%m-%d")  
    ax1.xaxis.set_major_formatter(date_format)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
    ax1.set_title(title)
    ax1.legend(labels = a.columns,loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
#    fig.autofmt_xdate()
    plt.show()
  
    
    
#allow for typing in 
codegrp = input('请输入您想查询的股票代码并以逗号分隔\n')
sel = input('请输入基本单位，查看每天数据输入‘d’,若为高频数据输入‘m’\n')  # \n为转行
print('友情提醒：输入的时间范围为20141208-20161123')
s_time = input('请输入起始时间，格式为20150921\n')
f_time = input('请输入结束时间，格式同上\n')
codegrp = codegrp.split(',')



#sample for test
#codegrp = ["000010","000008"]
#sel = "d"
#s_time = "20161008"
#f_time = "20161101"



start_time = s_time[0:4] + '-' + s_time[4:6] + '-' + s_time[6:]
final_time = f_time[0:4] + '-' + f_time[4:6] + '-' + f_time[6:]

if sel == 'd':
    start_time = pd.to_datetime(start_time + ' ' + '15:00:00')
    final_time = pd.to_datetime(final_time + ' ' + '15:00:00')
    time_range = pd.date_range(start_time, final_time)
    title ='Stocks:'+ ' ' + s_time + ' ' + 'to ' + ' ' + f_time + ' ' + 'Trend in day'
    
else:
    start_time = pd.to_datetime(start_time + ' ' + '09:25:00')
    final_time = pd.to_datetime(final_time + ' ' + '15:00:00')
    time_range = pd.date_range(start_time, final_time, freq='Min') 
    title =  'Stocks:'+ ' ' + s_time + ' ' + 'to' + ' ' + f_time + ' ' + 'Trend in minute'
  
    
#merge stocks data into one dataframe for plotting    
df = pd.DataFrame()

for code in codegrp:
    if code[0] == '6':
        code += '.SH'
    else:
        code += '.SZ'

    print(code)
    assert os.path.exists('smdata/' + code + '.txt'), 'No Stock'

    
    df_tmp = pd.read_csv('smdata/' + code + '.txt', parse_dates=[0]).sort_values('time', ascending=True)
    df_tmp.index = df_tmp['time']
    
    a = df_tmp.loc[df_tmp.time.isin(time_range), ['close']]
    a.rename(columns = {'close' : code[:-3]},inplace = True)
    df = pd.concat([df,a],axis=1) 


mpl.rcParams['font.sans-serif'] = [u'Kaiti']  # SimHei 黑体；Kaiti 楷体
mpl.rcParams['axes.unicode_minus'] = False
mplot(df,title)
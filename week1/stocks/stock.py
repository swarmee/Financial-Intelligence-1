# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:26:10 2017

@author: X240
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:52:24 2017

@author: Qing Li
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import os
from datetime import *


def mplot(a, title):
    x = list(range(len(a.time)))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x, a.close, 'b-', linewidth=2)
    ax1.plot(x, a.close, 'yo', markersize=2)

    def format_fn(tick_val, tick_pos):
        if int(tick_val) in x:
            return a['time'][a.index[int(tick_val)]]
        else:
            return ''

    ax1.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=30, integer=True))
    ax1.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# code = input('请输入您想查询的股票代码\n')
# sel = input('请输入基本单位，查看每天数据输入‘d’,若为高频数据输入‘m’\n')  # \n为转行
# print('友情提醒：输入的时间范围为20141208-20161123')
# s_time = input('请输入起始时间，格式为20150921\n')
# f_time = input('请输入结束时间，格式同上\n')


code = "000010"
sel = "d"
s_time = "20161008"
f_time = "20161123"


if code[0] == '6':
    code += '.SH'
else:
    code += '.SZ'

print(code)
assert os.path.exists('smdata/' + code + '.txt'), 'No Stock'

df = pd.read_csv('smdata/' + code + '.txt', parse_dates=[0]).sort_values('time', ascending=True)
start_time = s_time[0:4] + '-' + s_time[4:6] + '-' + s_time[6:]
final_time = f_time[0:4] + '-' + f_time[4:6] + '-' + f_time[6:]

matplotlib.rcParams['font.sans-serif'] = [u'Kaiti']  # SimHei 黑体；Kaiti 楷体
matplotlib.rcParams['axes.unicode_minus'] = False

if sel == 'd':
    start_time = pd.to_datetime(start_time + ' ' + '15:00:00')
    final_time = pd.to_datetime(final_time + ' ' + '15:00:00')
    time_range = pd.date_range(start_time, final_time)
    # print(df.time.isin(time_range))
    a = df.loc[df.time.isin(time_range), ['time', 'close']]


    def change_time(time):
        return time.strftime('%Y-%m-%d')


    a.time = a.time.apply(change_time)
    mplot(a, 'Stock' + ' ' + code[:-3] + ':' + s_time + 'to ' + f_time + 'Trend in day')

else:
    start_time = pd.to_datetime(start_time + ' ' + '09:25:00')
    final_time = pd.to_datetime(final_time + ' ' + '15:00:00')
    time_range = pd.date_range(start_time, final_time, freq='Min')
    a = df.loc[df.time.isin(time_range), ['time', 'close']]
    mplot(a, 'Stock' + ' ' + code[:-3] + ':' + s_time + 'to' + f_time + 'Trend in minute')


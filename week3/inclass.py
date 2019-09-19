## -*- coding: utf-8 -*-
#"""
#Created on Mon Mar 13 15:21:24 2017
#
#@author: Hou Lan
#"""
#

import numpy as np #index & value
import pandas as pd
import matplotlib as plt
from pandas import Series, DataFrame

#arr = np.array([1,2,3,4,5,6])
#print(arr.dtype)
#print(arr)
#
#float_arr = arr.astype(np.float64)
#print(float_arr)
#
#arr = np.ones(10)#10 one in 1 array
#arr = np.zeros(10) #10 zero in 1 array
#arr = np.empty((2,4,2))#get empty array
#int_array = np.arange(10)#get int number from zero to nine
#
##array适用于矩阵的点乘
#arr = np.array([[1,2,3],[4,5,6]])
#print(arr-arr)
#print(1/arr)
#print(arr*arr) #对应元素相乘
#print('arr*0.5=',arr*0.5)
#
##pyhton的左闭右开,零打头的情结
#arr = np.arange(10)
#print(arr)
#print(arr[5])
#print(arr[5:8])
#new_arr = arr[5:8].copy()
#new_arr[0] = 11
#print(arr,new_arr)
#
#arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
#print(arr2d)
#print(arr2d[:2])
#print(arr2d[:2,1:])#first two rows and first col
#arr2d[:2,1:] = 0 #array can be 
#
#names = np.array(['joe','bob','will','bob','will','joe','joe'])
#data = np.random.randn(7,4)
#print(data[data>1])
#print("bob data=",data[names=='bob'])
#print("!bob data=",data[names!='bob'])
#print("Partial bob data=",data[names=='bob',2:])
#print('data<0',data[data<0])
#data[names!='bob']=0
#
#
#
#arr = np.empty((8,4))
#for i in range(8):
#    arr[i] = i
#print(arr[[4,3,0,6]])#print the fourth, thirdth, first, sixth rows
#print(arr[[-2,-1]])#Using neg indices select rows from the end
#     
#      
#arr = np.arange(32).reshape(8,4)
#print(arr[np.ix_([1,5],[0,2])])
#print(arr.T) #转置
#
#arr = np.arange(8)+1
#print(np.square(arr))
#
#arr = np.random.rand(4,4)
#print(np.where(arr>0,0,arr))
#
#alist = list(range(10))
#result = [(2 if c>0 else c) for c in alist]
#
#
#alist = [1,2,3,4,5]
#result = [(0 if c>2 else c) for c in alist]
#result = np.reshape(result,(2,4))
##list 与 array 的转换 list(arr) 
#print('alist \n',result)

#对原来的list进行改
#alist = [1,2,3,4,5]
#for c in range(len(alist)):
#    if alist[c]>2:
#        alist[c] = 0
#        
#print(alist)
#
#print(np.where(arr>2,0,arr))
obj = Series([4,7,-5,3])
print(obj.values)
print(obj.index)
obj2 = Series([4,7,-5,3],index=['d','b','a','c'])
print(obj.index)
obj2['a']
obj2['d'] = 6
obj[['c','a','d']]#choose according to index
obj2[obj2>0]
#用seires改dataframe的列

#******************build dataframe******************
data = {"state":["ohio","ohio","ohio","nevada","nevada"],"year":[2000,2001,2002,2001,2002],"pop":[1.5,1.7,3.6,2.4,2.9]}
frame = pd.DataFrame(data)#可设定index,columns
#frame["marker"] = frame.state=="ohio"

frame.notnull().any()
frame.isnull().any()
#series 也用同样方法
















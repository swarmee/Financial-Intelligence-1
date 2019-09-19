# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:46:41 2017

@author: Hou Lan
"""

#************math module************
#import math
#
#math.sqrt(9)
#3*(2+6)
##2/0 ZeroDivisionError: division by zero
#
##************ 赋值  ************
#x = 3
#print(type(x))
#x = 'hello world'    
#isinstance(x,str) #return True or False
#
#x = 3 
#print(x**2)#幂运算
#
#x+=6 #x = x+6
#print(x)
#
#x = [1,2,3] #new list
#x[1] = 5 # change the value
#print(x) #print out list
#print(x[2]) #print out one value
#
#x = (1,2,3)
#print(x)
##x[1] = 5 TypeError: 'tuple' object does not support item assignment
#
#x = 3
#id(x)
#y = x
#id(y)
#
#x += 6
#id(x)
#id(y)
#
#x = [1,1,1,1]
#id(x[0]) = id(x[1]) #return True
#
##use del to delete unnecessary value
#
#a = 'abc' + '123' #merge string
#'{0}{1}{0}'.format('abra','cad')#formatize
#
#a = 3.6674
#'%5.3f' % a #字符串总共保留至少5位，小数点后3位
#
#"%d:%c"%(65,65) #get '65:A'
#
#"""My name is %s, and my age is %d"""%('Dong Fuguo',39)
#
#print('Hello\nWorld')
#print('\101')#三位八进制对应的字符
#print('\x41')#两位十六位进制对应的字符
#
#     
#path = r'C:\Windows\notepad.exe'
#print(path)
#
#
#3//5 #求整商
#3/5
#3.0/5
#3.0//5
#-13//10 # return -2 不超过-1.3
#
#
#[1,2,3] + [4,5,6]
#(1,2,3)+(4,)
#'abcd' + '1234'
#'A' + 1 #TypeError: Can't convert 'int' object to str implicitly
#True + 3
#False + 3
#
#2.0 * 3
#"a" * 10
#[1,2,3] * 3
#(1,2,3) * 3
#
#3.1%2 #return 1.1
#5.7%4.8#return 0.9000000000000004
#
#1<3<5 #return True
#'Hello'>'World' #return False
#[1,2,3]<[1,2,4]
#'Hello'>3#TypeError: unorderable types: str() > int()
#{1,2,3}<{1,2,3,4}#测试是否子集
#
#3 in [1,2,3]
#5 in range(1,10,1)
#'abc' in 'abcdefg'
#for i in (3,5,7):
#    print(i,end='\t')
#
#
##identify comparison about memory address
#3 is 3
#x = [300,300,300]
#x[0] is x[1] #return True
#x = [1,2,3]
#y = [1,2,3]
## return false cuz list x and list y are not comparable 
#
##set
#{1,2,3}|{3,4,5}# return {1, 2, 3, 4, 5}
#{1,2,3}&{3,4,5}# return  {3}
#{1,2,3}^{3,4,5}# return {1, 2, 4, 5}
#{1,2,3}-{3,4,5}# return  {1, 2}
#
##**************************numpy module****************************
#import numpy as np
#x = np.ones(3)
#m = np.eye(3)*3
#m[0,2] = 5
#m[2,0] = 3
#x@m #matric multiply
#
#
##*****************************get function*************************
#dir(__builtins__)
#
#
##*********************ord() and chr() function*********************
#ord('a')
#chr(65)
#chr(ord('A')+1) 
#str(1)
#str(1234)
#str([1,2,3])
#str({1,2,3})
#str((1,2,3))
#
#
##************************max() min() sum()************************
#import random
#a = [random.randint(1,100) for i in range(10)]
#
#print(max(a),min(a),sum(a))
#sum(a)/len(a) # get average
#
#x = ['21','1234','9']
#max(x,key=len)#用key指定规则
#max(x,key=int)
#
#
#
##************************type() isinstance()************************
#type([3])
#type({3}) in (list,tuple,dict)
#isinstance(3,int)
#isinstance(3j,(int,float,complex))
#
#x = ['aaaa','bc','d','b','ba']
#sorted(x,key=lambda item:(len(item),item))
#
#
#
#
##************************range() - [start,end)************************
#list(zip('abcd',[1,2,3]))
#
#x = [1,2,3,4,5,6]
#y = 3
#z = y
#print(y)
#del y
#print(y)#NameError: name 'y' is not defined
#print(z)
#
#del x[1]
#print(x)
#
#x = (1,2,3)
#del x[1]#TypeError: 'tuple' object doesn't support item deletion
#
#x = input('提示：')
#
##************************open()************************
#fp = open(r'C:\Users\X240\Desktop\test.txt','a+')
#print('Hello,world!',file = fp)
#fp.close()
#
#for i in range(10,20):
#    print(i,end=' ')
#    
#    
##************************math module************************
#import math
#math.sin(0.5)
#import random
#x = random.random() #获得[0,1）内的随机小树
#n = random.randint(1,100)#获得[1,100)上的随机整数
#
#from math import sin
#sin(3)
#from math import sin as f
#f(3)
#
#
#
#
##************************三位自然数，计算个十百上的数字************************
##method 1
#x = input("请输入一个三位数：")
#x = int(x) #注意输入是字符
#a = x//100
#b = x//10%10
#c = x%10
#print('百：',a,'十：',b,'个：',c)
##method 2
#x = input("请输入一个三位数：")
#x = int(x) #注意输入是字符
#a,b = divmod(x,100)
#b,c = divmod(b,10)
#print(a,b,c)
##method 3
#x = input("请输入一个三位数：")
#a,b,c = map(int,x)
#print(a,b,c)
#
#
#
#
#
#
#
##************************calculate efficiency************************
#import time
#
#result = [] 
#start = time.time()
#for i in range(10000):
#    result = result + [i]
#
#print(len(result),',',time.time()-start)
#
#result = []
#start = time.time()
#for i in range(10000):
#    result.append(i)
#    
#print(len(result),',',time.time()-start)
#
#
#
#
#
##************************remove************************
#
#x = [1,2,1,2,1,1,1]
##错误
#for i in x[::]:
#    if i ==1:
#        x.remove(i)
#    print(x)
#
#
##************************List pieces************************
#aList = [3,5,7]
#aList[len(aList):]=[9]
#aList[:3] = [1,2,3]
#aList[:3] = []
#aList = list(range(10))
#aList[::2] = [0]*5                     
#
#
##************************List sort************************
#aList = [3, 4, 5, 6, 7, 9, 11, 13, 15, 17]
#import random
#random.shuffle(aList)
#print(aList)
#
#aList.sort()#默认升序
#aList.sort(reverse = True)                                                                     
#aList.sort(key = lambda x:len(str(x))) #按转换成str的长度排序
#
#sorted(aList)#升序排列并返回新列表
#sorted(aList,reverse = True)#降序
#
#      
##************************List sum************************
#sum(range(1, 11))   #sum()函数的start参数默认为0
#sum(range(1, 11), 5)#指定start参数为5，等价于5+sum(range(1,11))
#
##************************List************************
#aList = [x*x for x in range(10)]
#aList = []
#for x in range(10):
#    aList.append(x*x)
#
#aList = list(map(lambda x: x*x,range(10)))
#
##************************dict****************************
#scores = {"Zhang San": 45, "Li Si": 78, "Wang Wu": 40, "Zhou Liu": 96, "Zhao Qi": 65, "Sun Ba": 90, "Zheng Jiu": 78, "Wu Shi": 99, "Dong Shiyi": 60}
#highest = max(scores.values())
#lowest = min(scores.values())
#average = sum(scores.values())*1.0/len(scores)
#highestPerson = [name for name, score in scores.items() if score == highest]
#
#
#
##************************tuple****************************
#g = ((i+2)**2 for i in range(10))
#g.__next__()
#
#
##************************dict****************************
##使用dict利用已有数据创建字典
#keys = ['a','b','c','d']
#values = [1,2,3,4]
#dictionary = dict(zip(keys,values))
##使用dict根据给定的键、值创建字典
#d = dict(name='Dong',age = '37')
##以给定内容为键，创建值为空的字典
#aDict = dict.fromkeys(['name','age','sex'])
#print(aDict.get('a'))
#aDict['score'] = aDict.get('Score',[])
#aDict['score'].append(98)
##使用字典对象的items()方法可以返回字典的键、值对列表
#aDict={'name':'Dong', 'sex':'male', 'age':37}
#for item in aDict.items():
#    print(item)
#for key in aDict:
#    print(key)  
#for key,value in aDict.items():
#    print(key,value)
##使用字典对象的keys()方法可以返回字典的键列表
#aDict.keys()
##使用字典对象的values()方法可以返回字典的值列表
#aDict.values()
##使用update添加与修改
#aDict.update({'a':'a','b':'b'})
##使用del删除字典中指定键的元素
#
##使用字典对象的clear()方法来删除字典中所有元素
#
##使用字典对象的pop()方法删除并返回指定键的元素
#
##使用字典对象的popitem()方法删除并返回字典中的一个元素

#******首先生成包含1000个随机字符的字符串，然后统计每个字符的出现次数**********
#import string 
#import random
#
#x = string.ascii_letters + string.digits + string.punctuation
#y = [random.choice(x) for i in range(1000)]
#z = ''.join(y)
#d = dict()
#for ch in z:
#    d[ch] = d.get(ch,0) + 1
     
#***********************************set******************************************
#直接将集合赋值给变量
#a = {3,5}
#a.add(7)

##使用set将其他类型数据转换为集合
#a_set = set(range(8,14))
##自动去除重复
#b_set = set([0, 1, 2, 3, 0, 1, 2, 3, 7, 8])     
##空集合
#c_set = set()                                              
##使用del删除整个集合

#a = {1,4,3,2,3}
#a.pop()
#a.add(2)
##移除所有3
#a.remove(3)
#a.clear()

#a_set = set([8, 9, 10, 11, 12, 13])
#b_set = {0, 1, 2, 3, 7, 8}
#print(a_set|b_set)#并集
#print(a_set&b_set)#交集
#print(a_set-b_set)#差集
#{3}.issubset({3,4})       #测试是否为子集
#{3}.isdisjoint({4}) #如果两个集合的交集为空，返回True
##集合包含关系测试">(=)","<(=)"
                              

#import random
#import time
#
#def RandomNumbers(number, start, end):
#    '''使用列表来生成number个介于start和end之间的不重复随机数'''
#    data = []
#    n = 0
#    while True:
#        element = random.randint(start, end)
#        if element not in data:
#            data.append(element)
#            n += 1
#        if n == number - 1:
#            break
#    return data
#
#def RandomNumbers1(number, start, end):
#    '''使用列表来生成number个介于start和end之间的不重复随机数'''
#    data = []
#    while True:
#        element = random.randint(start, end)
#        if element not in data:
#            data.append(element)
#        if len(data) == number:
#            break
#    return data

#最高效
#def RandomNumbers2(number, start, end):
#    '''使用集合来生成number个介于start和end之间的不重复随机数'''
#    data = set()
#    while True:
#        data.add(random.randint(start, end))
#        if len(data) == number:
#            break
#    return data

#start = time.time()
#for i in range(100):
#    RandomNumbers(1000, 1, 10000)
#print('Time used:', time.time()-start)
#
#start = time.time()
#for i in range(100):
#    RandomNumbers1(1000, 1, 10000)
#print('Time used:', time.time()-start)

#start = time.time()
#for i in range(100):
#    RandomNumbers2(1000, 1, 10000)
#print('Time used:', time.time()-start)
#
#*******************************循环****************************************
#x = input('input two number')#以空格隔开
#a,b = map(int,x.split())
#if a>b:
#    a,b = b,a
#print(a,b)

#chTest = ['1', '2', '3', '4', '5']
#if chTest:
#	  print(chTest)
#else:
#	  print('Empty')
#

#if elif
#尽量将计算提前
#digits = (1,2,3,4)
#for i in range(1000):
#    result = []
#    for i in digits:
#        i = i*100
#        for j in digits:
#            j = j*10
#            for k in digits:
#                result.append(i+j+k)
#计算小于100的最大素数
#for n in range(100,1,-1):
#    for i in range(2,n):
#        if n%i ==0:
#            break
#    else:
##        print(n)
##        break
#        print(n,end=' ')

#计算1+2+3+…+100 的值
#s = 0
#for i in range(1,101):
#    s = s + i
#print('1+2+3+…+100 = ', s)
#print('1+2+3+…+100 = ', sum(range(1,101)))
##求平均分
#score = [70, 90, 78, 85, 97, 94, 65, 80]
#s = 0
#for i in score:
#	s += i
#print(s/len(score))
#print(sum(score) / len(score))                 #也可以直接这样做
#


s="apple,peach,banana,peach,pear"
#s.find("peach")
#s.find("peach",7)
#s.find("peach",7,20)
#s.rfind('p')
#s.index('p')
#s.index('pe')
#s.count('p')

#li = s.split(',')
#s.partition(',')
#s.rpartition(',')
#s = "2014-10-31"
#t=s.split("-")
#print(t)
#print(list(map(int, t)))
#s.lower()
#s.capitalize()
#s.title()
#s.swapcase()                
#s.replace
#
#table = ''.maketrans('abcdef123', 'uvwxyz@#$')
#s = "Python is a greate programming language. I like it!"
#s.translate(table)
##s.strip() s.rstrip() s.lstrip()
#
#import string
#string.ascii_letters
#string.punctuation
#string.ascii_lowercase
#string.ascii_uppercase

def demo(newitem,old_list=[]):
    old_list.append(newitem)
    return old_list
print(demo('5',[1,2,3,4]))   #right
print(demo('aaa',['a','b'])) #right
print(demo('a'))             #right
print(demo('b'))             #wrong
































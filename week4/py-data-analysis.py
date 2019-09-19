# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:56:39 2017 for Python Data Analysis

@author: Hou LAN

"""

def add_and_maybe_multiply(a,b,c=None):
    result = a + b
    if c is not None:
        result = result*c
    return result
# None不是关键字，只是NoneType的一个实例

#*******************datetime************************
from datetime import datetime,date,time
dt = datetime(2011,10,29,20,30,21)
#注意dt.date()和dt.time()的区别

print(dt.strftime('%m%d%Y %H:%M'))#格式化datetime为字符串
print(datetime.strptime('20091031','%Y%m%d'))#字符串转化为datetime对象
print(dt.replace(minute=0,second=0))
dt2 = datetime(2011,11,15,22,30)
delta = dt2-dt #两个datetime的差会产生一个datetime.timedelta

#*******************zip 与 enumerate************************
seq1 = ['foo','bar','baz']
seq2 = ['one','two','three']
for i, (a,b) in enumerate((zip(seq1,seq2))):
    print('%d: %s, %s' % (i,a,b))


#*******************dict************************
words = ['apple','bat','bar','atom','book']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)
#   by_letter.setdefault(letter,[]).append(word)

    
from collections import defaultdict
by_letter = defaultdict(list)
for word in words:
    by_letter[word[0]].append(word)
    
    
#*******************set************************
a = {1,2,3,4,5}
b = {3,4,5,6,7,8}

a|b #并
a&b #交
a-b #差
a^b #异或

a_set = {1,2,3,4,5}
{1,2,3}.issubset(a_set)
a_set.issuperset({1,2,3})



#*******************list,set,dict************************
#[expr for val in collection if condition]
#dict_comp = {key-expr: value-expr for value in coolection if condition}
#set_comp = {expr for value in collection if condition}


all_data = [['Tom','Billy','Jefferson','Andrew','Wesley','Steven','Joe'],['Susie','Casey','Jill','Ana','Eva','Jennifer','Stephanie']]
[name for names in all_data for name in names if name.count('e')>=2]#注意names 和 name循环的位置 先大范围再小范围



import re
states = ['  Alabama','eorgial!','Georgia']
def remove_punctuation(value):
    return re.sub('[!#?]','',value)

clean_ops = [str.strip,remove_punctuation, str.title]
#title()词语的第一个字母大写
def clean_strings(strings,ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result


    
    
    
    
    
    
    
    
    
    
    
    
    
    
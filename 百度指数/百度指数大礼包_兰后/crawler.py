# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:18:21 2017

@author: Hou LAN 41423018
@github: https://github.com/workinDead/index_crawler
"""
import time
import os
import glob # pip install glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.externals import six
from matplotlib.ticker import FuncFormatter, MaxNLocator
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait #pip install selenium and remember to download chromedriver.exe

from PIL import Image # pip install pillow
from pyocr import tesseract # pip install pyocr



chrome_path = r"C:\Users\X240\Desktop\chromedriver.exe"
option_path = r'C:\Users\X240\AppData\Local\Temp\scoped_dir9624_1152'# 获取用户设置数据地址 请根据自身的用户数据地址更改本地址 #chrome://version/ 
# 若不愿查找地址，可解锁模拟登陆注释

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('enable-automation')
chrome_options.add_argument('--user-data-dir='+os.path.abspath(option_path))
driver = webdriver.Chrome(chrome_options=chrome_options,executable_path=chrome_path)
#driver.implicitly_wait(60) # 隐性等待30秒  

driver.maximize_window()
url = r'https://index.baidu.com'
driver.get(url)

#打开新的标签页
#js = 'window.open("http://index.baidu.com");'
#driver.execute_script(js)
#handles = driver.window_handles
#driver.switch_to_window(handles[-1])

#获取缓存
#driver.get_cookies()
#driver.find_element_by_xpath("""//*[@id="userbar"]/ul/li[4]/a""").click()
#emailFieldElement = WebDriverWait(driver,10).until(lambda driver : driver.find_element_by_xpath("""//*[@id="TANGRAM_12__userName"]"""))
#passFieldElement = WebDriverWait(driver,10).until(lambda driver : driver.find_element_by_xpath("""//*[@id="TANGRAM_12__password"]"""))

#输入用户名和密码
#userName = "请输入用户名"
#password = "请输入密码"

#模拟登陆
#emailFieldElement.clear()
#emailFieldElement.send_keys(userName)
#passFieldElement.clear()
#passFieldElement.send_keys(password)
#driver.find_element_by_xpath("""//*[@id="TANGRAM_12__submit"]""").click()

#获取用户输入关键词
searchText = input('请输入需要搜索的关键词：')
searchFieldElement = WebDriverWait(driver,10).until(lambda driver:driver.find_element_by_xpath("""//*[@id="schword"]"""))
searchFieldElement.clear()
searchFieldElement.send_keys(searchText)
searchFieldElement.send_keys(Keys.ENTER)

#向下滑动页面
#time.sleep(2)
#driver.find_element_by_xpath("""/html/body""").send_keys(Keys.PAGE_DOWN)

#找到图形框
time.sleep(3)
xoyelement = driver.find_elements_by_css_selector("#trend rect")[2]
presentFilepath = os.getcwd() #获取当前文件位置
imgFilepath = ''.join([presentFilepath,'\\',searchText])

#如果不存在，则在本运行文件中创建一个相应文件夹
if os.path.exists(imgFilepath)==False:
    os.mkdir(imgFilepath)
    


# 得到关键词的字节长度
length = len(searchText)
utf8_length = len(searchText.encode('utf-8'))
length = (utf8_length - length)/2 + length

# 是否 变更时限

a = input('确认时限输入Y.如果需要重新选择请在输入本栏前点击所选时限:')

    
dayElement = driver.find_element_by_class_name('chartselect-click')
duration = int(dayElement.text.strip('天'))
# 坐标初始化
x_0 = 1
y_0 = 0
try:

    
    for i in range(duration):
        
    
        #模拟鼠标悬浮
        ActionChains(driver).move_to_element_with_offset(xoyelement,x_0,y_0).perform()
        if duration == 7:
            x_0 = x_0 + 202.33
        elif duration == 30:
            x_0 = x_0 + 41.68
        elif duration == 90:
            x_0 = x_0 + 13.64
        elif duration == 180:
            x_0 = x_0 + 6.78
        
        
        time.sleep(3)
        imgelement = driver.find_element_by_xpath('//div[@id="viewbox"]')
        
        #找到图片坐标
        locations = imgelement.location
        print(locations)
        
        #找到图片大小
        sizes = imgelement.size
        print(sizes)
        
        #初始化位置
        init_pos = 120
        #定位指定图片的位置
        left = locations['x']
        top = locations['y']
        right = locations['x'] + sizes['width']
        bottom = locations['y'] + sizes['height']
        
        #保存图片
        imgName = ''.join([str(i + 1),'.png'])
        driver.save_screenshot(imgName)
        #打开图片
        img = Image.open(imgName)
        #剪裁图片
        img = img.crop((left,top-init_pos,right,bottom-init_pos))
        img.save(''.join([imgFilepath,'\\',imgName]))
        
        

except Exception as err:
    print(err)
    print(i+1)

########################################
#>>> 1> # Function to recognize image 
def rec_img(img):
    
    width = img.size[0]
    height = img.size[1]
    #构造指数的位置
    rangle = (24.5+6.1*length+5,int(height/2),int(width),int(height))  #左、上、右、下
    # 打开截图切割
    img = img.crop(rangle)
    # 将图片放大
    (x, y) = img.size
    x_s = int(x*2.4)
    y_s = int(y*2.4)
    imgzoom = img.resize((x_s,y_s),Image.ANTIALIAS)
    code = tesseract.image_to_string(imgzoom)
    result = re.sub("\D", "", code)
    return result


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

 
# 储存数字的字典
dict_indexes = {}     
for filepath in glob.glob(''.join([imgFilepath,'\\*.png'])):
    filename = filepath.split('\\')[-1].split('.')[0]
    img = Image.open(filepath)
    dict_indexes[filename] = rec_img(img)         


date = driver.find_element_by_xpath('//*[@id="auto_gsid_7"]/span[2]')

start_date = date.text.split('至')[0].strip()
end_date = date.text.split('至')[1].strip()
start_date = pd.to_datetime(start_date, format=r"%Y-%m-%d")
end_date = pd.to_datetime(end_date, format=r"%Y-%m-%d")
time_range = pd.date_range(start_date,end_date)
df = pd.DataFrame.from_dict(dict_indexes,orient='index')
df.index = df.index.astype('int')
df.sort_index(inplace=True)
df['date'] = time_range
df.rename(columns={0:'index'},inplace=True)
render_mpl_table(df)
  
  

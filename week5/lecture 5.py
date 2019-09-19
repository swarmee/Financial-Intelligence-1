# http://pandas-datareader.readthedocs.io/en/latest/remote_data.html
# http://blog.csdn.net/xtfge0915/article/details/52938740
import pandas_datareader.data as web
import datetime
import pandas as pd
# #################################################################################
# #
# #    write a file
# #
# #################################################################################
f = open("sample.txt","a+")
contents = "hello, word"
f.write(contents)
f.close()

# ##########################
try:
    f = open("sample.txt","w")
    contents = "Hello, world"
    f.write(contents)
except IOError as error:
    print(str(error))
finally:
    f.close()
# #############################
with open("sample.txt","a+") as out:
    contents = "hello, world"
    out.write(contents)
# ###############################

#dsffsdf
start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2017, 3, 26)
#
# Get AAPL stock prices and show it

AAPL = web.DataReader("AAPL","yahoo",start, end)
print(AAPL.head(5))
print(type(AAPL))
print(AAPL.tail(10))
print(AAPL["Adj Close"].head())
print(AAPL["Adj Close"].head())
print(AAPL.index)

try:
    AAPL.to_csv(r"C:\Users\kooli\Desktop\out.cvs")
except IOError as error:
    print(str(error))
    exit()


try:
    data = pd.read_csv(r"C:\Users\kooli\Desktop\out.cvs")
    print(type(data))
    print(data.tail(3))
except IOError as error:
    print(str(error))

AAPL = data

import matplotlib.pyplot as plt
import numpy as np
# # ***********************draw a simple graph ***************************************

np.random.seed(1000)
y = np.random.standard_normal(20)
x = range(len(y))
plt.plot(x,y)
plt.show()


# # *****************************draw stock  in one graph *****************************
#
plt.plot(AAPL.index,AAPL["Adj Close"],marker="o",linestyle = "dashed")
plt.show()
import pylab
pylab.rcParams['figure.figsize'] = (10, 5)
AAPL["Adj Close"].plot(grid = True)
plt.show()
# # ***********************************************************************************
# #
# # # *****************************draw stock in two graphs ******************************
fig = plt.figure()
ax_price = fig.add_subplot(1,2,1)
ax_volume = fig.add_subplot(1,2,2)

ax_price.plot(AAPL.index, AAPL["Adj Close"])
ax_price.set_xticklabels(AAPL.index, rotation = 30, fontsize="small")
ax_price.legend(loc = "best")

ax_volume.plot(AAPL.index,AAPL['Volume'])
ax_volume.set_xticklabels([str(x)[0:11] for x in AAPL.index], rotation = -30, fontsize="small")
ax_volume.set_title("AAPL Volume Trends")
ax_volume.legend(loc = "best")
plt.show()
# # ***************************************************************************************
#
# # *****************************draw stocks **********************************************

all_data = {}

for ticker in ['AAPL','IBM','GOOG']:
    all_data[ticker] = web.DataReader(ticker, 'yahoo', start, end)

print(all_data)

import pickle

f = open(r"stocks.txt","wb")

try:
    pickle.dump(all_data,f)
except IOError as error:
    print(str(error))
finally:
    f.close()


f = open(r"stocks.txt", "rb")
try:
    all_data = pickle.load(f)
except IOError as error:
    print(str(error))



# print(all_data)

from pandas import DataFrame

price = DataFrame({tic: data['Adj Close'] for tic, data in all_data.items()})
volume = DataFrame({tic: data['Volume'] for tic, data in all_data.items()})

# print("price = \n", price.tail(5))
# print("price AAPL= \n", price['AAPL'])
# print("price AAPL= \n", price.AAPL)
# print("volume = \n", volume.tail(5))


import matplotlib.pyplot

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(price.index,price['AAPL'], linestyle= "--", label="apple")
ax.plot(price['IBM'].index,price['IBM'].values, linestyle=  '-.' ,label="IBM")
# ax.plot(price['GOOG'].index,price['GOOG'].values)
ax.legend(loc = "best")
plt.show()

print(type(price))
# price.plot()
price["IBM"].plot()
plt.show()

# # ***************************************************************************************
#
returns = price.pct_change()
print(returns.tail(10))

# # The corr method of Series computes the correlation of the overlapping, non-NA,
# # aligned-by-index values in two Series. Relatedly, cov computes the covariance:
cov = returns.AAPL.corr(returns.IBM)
print(cov)

# # DataFrame’s corr and cov methods, on the other hand,
# # return a full correlation or covariance matrix as a DataFrame, respectively:

print(returns.corr())
cor = returns.corr()
print(cor.idxmin())


print(returns.cov())

# # Using DataFrame’s corrwith method, you can compute pairwise correlations between a DataFrame’s columns or rows with another Series or DataFrame.
# # Passing a Series returns a Series with the correlation value computed for each column:
aSeries = returns.corrwith(returns.GOOG)
print("GOOG\n",aSeries)
aSeries.order()
print("GOOG\n",aSeries.order(ascending=False))

# #Passing a DataFrame computes the correlations of matching column names.
# # Here I compute correlations of percent changes with volume:
print("returns with volume\n",returns.corrwith(volume))
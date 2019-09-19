# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 07:09:30 2017

@author: Hou Lan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn import cross_validation
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.grid_search import GridSearchCV
from keras.optimizers import SGD


#*********************************************#
#
#    Data preprocessing
#
#*********************************************#

########################################
#>>> 1>  Function for Splitting dataset into trainset and testset

def train_test_split(X,y,test_size=0.05):
    
    test_num = int(np.ceil(test_size * len(X)))
    X_train = X[:-test_num]
    X_test = X[-test_num:]
    y_train = y[:-test_num]
    y_test =  y[-test_num:]
    return X_train,X_test,y_train,y_test


#*********************************************#
#
#    SVM
#
#*********************************************#


########################################
#>>> 1>  Function for Using SVM to predict

def svr_predict_aids(X_train,y_train,dates,X,y,x):
    """
    Visualize applying SVM
    """
    # Build model
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
    svr_rbf.fit(X_train,y_train)
    svr_lin = SVR(kernel= 'linear', C= 1e3)
    svr_lin.fit(X_train,y_train)
    svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
    svr_poly.fit(X_train,y_train)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0,len(date_tick))
    
    ax.scatter(np.reshape(list(range(len(date_tick))),(len(date_tick), 1)), y, color= 'black', label= 'Data') # plotting the initial datapoints 
    ax.plot(range(len(X)),svr_rbf.predict(X), color= 'red', label= 'RBF model') # plotting the line made by rbf kernel
    ax.plot(range(len(X)),svr_lin.predict(X), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
    ax.plot(range(len(X)),svr_poly.predict(X), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
    ax.set_xlabel('Date')
    ax.set_ylabel('The change rate of cases of AIDS')
    ax.set_title('Support Vector Regression')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))

    dates = list(date_tick)
    dates.append(dates[-1])
    dates = np.array(dates)
    ax.set_xticklabels(dates.reshape(int(len(dates)/4),4)[:,0])
    
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    score_rbf = mean_squared_error(svr_rbf.predict(X_test),y_test)
    score_lin = mean_squared_error(svr_lin.predict(X_test),y_test)
    score_poly = mean_squared_error(svr_poly.predict(X_test),y_test)
    return score_rbf, score_lin, score_poly



#*********************************************#
#
#    Back propagation neural network
#
#*********************************************#

########################################
#>>> 0> # Function to view results
def show_best_score(grid_result):
    """
    print best choice for keras regression 
    """
    mse_arr = np.array([])
    for params, mean_score, scores in grid_result.grid_scores_:
        
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
        mse_arr = np.concatenate((mse_arr,np.array([scores.mean(),params])))
    
    mse_arr = mse_arr.reshape((int(len(mse_arr)/2),2)) # reshape to 2 dimensions for calculating mse
    print("Best: %f using %s" % (mse_arr[mse_arr[:,0]==mse_arr[:,0].min()][0][0],mse_arr[mse_arr[:,0] ==mse_arr[:,0].min()][0][1]))    



########################################
#>>> 1> # Function to create model for tuning activation
def create_model_1(activation = 'sigmoid'):
    
    # create model
    model = Sequential()
    model.add(Dense(30,input_dim=26,activation='relu'))
    model.add(Activation('relu'))    
    model.add(Dense(1,activation=activation))
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='SGD')
    return model



########################################
#>>> 2> # Function to create model for tuning optimal batchsize and training epochs
def create_model_2(batch_size = 30,input_dim = 26):
    
    # create model
    model = Sequential()
    model.add(Dense(batch_size,input_dim=input_dim,activation='relu'))
    model.add(Activation('relu'))    
    model.add(Dense(1,activation='tanh'))
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='SGD')
    return model


########################################
#>>> 3> # Function to create model for tuning optimizer
def create_model_3(optimizer = 'SGD'):
    
    # create model
    model = Sequential()
    model.add(Dense(30,input_dim=26,activation='relu'))
    model.add(Activation('relu'))    
    model.add(Dense(1,activation='tanh'))
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model



########################################
#>>> 4> # Function to create model for tuning learning rate and momentum

def create_model_4(learn_rate=0.01, momentum=0.9):
    
    # create model
    model = Sequential()
    model.add(Dense(30,input_dim=26,activation='relu'))
    model.add(Activation('relu'))    
    model.add(Dense(1,activation='tanh'))
    
    # Compile model
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model



########################################
#>>> 5> # Function to create model for tuning network weight initialization

def create_model_5(init_mode='uniform'):
    
    # create model
    model = Sequential()
    model.add(Dense(30,input_dim=26,kernel_initializer=init_mode,activation='relu'))
    model.add(Activation('relu'))    
    model.add(Dense(1,activation='linear'))
    
    # Compile model
    optimizer = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model




#*********************************************#
#
#       Visualization
#
#*********************************************#

########################################
#>>> 1> # Function to visualize correlation in heatmap 
def plot_corr(data):

    # set appropriate font and dpi
    sns.set(
            font='SimHei',  # fix Chinese output
            font_scale=1.2,
            rc={'axes.unicode_minus':False} )# fix negative notation output)
    sns.set_style({"savefig.dpi": 100})
    # plot it out
    ax = sns.heatmap(data, cmap=plt.cm.Blues, linewidths=.1)
    # set the x-axis labels on the top
    ax.xaxis.tick_top()
    # rotate the x-axis and y-axis labels
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # get figure (usually obtained via "fig,ax=plt.subplots()" with matplotlib)
    fig = ax.get_figure()
    # specify dimensions and save





if __name__ == '__main__':


    # Read excel file into dataframe
    df = pd.read_excel('final_index.xlsx')

    # Check data
    print(df.info())

    # Get correlation
    data = df.ix[:,1:].corr(method='pearson')
    
                     
          
    # Get the percent change
    column_names = df.columns     
    for name in column_names[1:]:
        df[name] = df[name].pct_change()
    
    # Drop the first row  without value
    df.drop(0,axis=0,inplace=True)
        
    
    # Standardize data 
    df_matrix = df.ix[:,1:].as_matrix()
    df_matrix = preprocessing.scale(df_matrix.T).T
    
    # Split dataset 
    X=df_matrix[:,1:]
    y=df_matrix[:,0]
    X_train,X_test,y_train,y_test=train_test_split(X,y,0.05)


    
    # Data preparation
    column_names = df.columns     
    date_tick = pd.to_datetime(df.ix[:,0].astype('str'), format=r"%Y.%m").dt.strftime('%Y-%m')  # convert type from float to datetime
        

                                                                         
    # SVM to predict
    score_rbf, score_lin, score_poly = svr_predict_aids(X_train,y_train,date_tick,X,y,len(y)-1)
    print("MSE of rbf :",score_rbf,'\n',"MSE of linear :",score_lin,'\n',"MSE of poly :",score_rbf)
    
    
    # BP Neural Network
    model = Sequential()
    
    # create model
    model.add(Dense(30,input_dim=26,activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Activation('relu'))    
    model.add(Dense(1,activation='linear'))
    #0.01-0.9
    # complie
    optimizer = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    # fit
    results = model.fit(X_train,y_train,epochs = 10, batch_size=10, verbose=0)
    score_bp = model.evaluate(X_test,y_test)
    print("MSE of testset IN BP: ",score_bp)
    
    # predict
    y_pred = model.predict(X)  
    
    # plot prediction data 
    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111)
    ax.set_xlim(0,len(date_tick))
    ax.scatter(range(len(y_pred)), y, color= 'black', label= 'Data') 
    ax.plot(range(len(y_pred)),y_pred,'b',label='Predict')
        
    ax.set_xlabel('Date')
    ax.set_ylabel('The change rate of cases of AIDS')
    ax.set_title('BP Neural Network')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))

    dates = list(date_tick)
    dates.append(dates[-1])
    dates = np.array(dates)
    ax.set_xticklabels(dates.reshape(int(len(dates)/4),4)[:,0])
    
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()  


#    # Find optimal activation
#    model = KerasRegressor(build_fn=create_model_1, epochs = 10, batch_size=10, verbose=0)
#    # define the grid search parameters
#    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#    param_grid = dict(activation=activation)
#    grid = GridSearchCV(estimator=model, param_grid=param_grid)
#    grid_result = grid.fit(X_train, y_train)
#    
#    show_best_score(grid_result)
    
#    # Find optimal batchsize and traning epoches
#
#    model = KerasRegressor(build_fn=create_model_2, verbose=0)
#    batch_size = [10,20,30]
#    epochs = [10,50]
#    
#    param_grid = dict(batch_size=batch_size,nb_epoch=epochs)
#    grid = GridSearchCV(estimator=model, param_grid=param_grid)
#    grid_result = grid.fit(X_train, y_train)
#    show_best_score(grid_result)
    
#    # Find optimaizer
#    model = KerasRegressor(build_fn=create_model_3, epochs = 10, batch_size=10, verbose=0)
#    # define the grid search parameters
#    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#    
#    param_grid = dict(optimizer=optimizer)
#    grid = GridSearchCV(estimator=model, param_grid=param_grid)
#    grid_result = grid.fit(X_train, y_train)
#    show_best_score(grid_result)
    
#    # Find optimal learning rate and momentum
#    model = KerasRegressor(build_fn=create_model_4, epochs = 10, batch_size=10, verbose=0)
#    # define the grid search parameters
#    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#    param_grid = dict(learn_rate=learn_rate, momentum=momentum)
#    grid = GridSearchCV(estimator=model, param_grid=param_grid)
#    grid_result = grid.fit(X_train, y_train)
#    
#    show_best_score(grid_result)




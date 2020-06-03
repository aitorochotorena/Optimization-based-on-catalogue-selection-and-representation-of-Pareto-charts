# import numpy package for arrays and stuff 
import numpy as np 

# import matplotlib.pyplot for plotting our result 
import matplotlib.pyplot as plt 

# import pandas for importing csv files 
import pandas as pd 

#decision tree-based models for regression 
from sklearn.tree import DecisionTreeRegressor

def average_DT(df_X,df_y,x_ref):
    # create a regressor object (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    regressorAver = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=1,
               max_leaf_nodes=len(df_X), min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               random_state=None, splitter='best')

    # fit the regressor with X and Y data 
    regressorAver.fit(df_X.values.reshape(-1,1) , df_y.values.reshape(-1,1)) 

    return regressorAver.predict(np.array([x_ref]).reshape(1, 1))[0]

def previous_DT(df_X,df_y,x_ref):
    # we call the x values of the dataframe
    xy=np.array(list(zip(df_X, df_y)))
    sorted_xy = np.unique(xy,axis=0)

    # a vector of supplementary points around the reference value to force the regression tree through X
    C=(np.vstack((sorted_xy[:,0]-sorted_xy[:,0].min()/1000,sorted_xy[:,0]+sorted_xy[:,0].min()/1000)).ravel('F'))
    D = np.repeat(C, 2)
    
    df_X = (np.delete(np.delete(D,2),2))
    
    #axis y
    df_y1 = sorted_xy[:,1]
    df_y1_C1 = df_y1-df_y1.min()/100
    df_y1_C2 = df_y1+df_y1.min()/100
    A=np.repeat(df_y1_C1, 2)
    B=np.repeat(df_y1_C2, 2)
    C=(np.vstack((A,B)).ravel('F'))
    df_y=(C[:-2])
    
    # select independent column
    df_XPrev = df_X.reshape(-1,1)

    # select dependent column
    df_yPrev = df_y.reshape(-1,1)

    # create a regressor object (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    regressorPrev = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=1,
               max_leaf_nodes=len(df_XPrev), min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               random_state=None, splitter='best')

    # fit the regressor with X and Y data 
    regressorPrev.fit(df_XPrev, df_yPrev) 
    
    return regressorPrev.predict(np.array([x_ref]).reshape(1, 1))[0]

def next_DT(df_X,df_y,x_ref):
    # we call the x values of the dataframe
    xy=np.array(list(zip(df_X, df_y)))
    sorted_xy = np.unique(xy,axis=0)
    C=(np.vstack((sorted_xy[:,0]-sorted_xy[:,0].min()/1000,sorted_xy[:,0]+sorted_xy[:,0].min()/1000)).ravel('F'))

    D=np.repeat(C, 2)
    D=D[:-2]
    df_X=D


    #axis y    
    
    df_y1 = sorted_xy[:,1]
    df_y1_C1 = df_y1-df_y1.min()/100
    df_y1_C2 = df_y1+df_y1.min()/100
    A=np.repeat(df_y1_C1, 2)
    B=np.repeat(df_y1_C2, 2)
    C=(np.vstack((A,B)).ravel('F'))
    C=(np.delete(np.delete(C,2),2))
    df_y=(C)

    
    # select independent column
    df_XNext = df_X.reshape(-1,1)

    # select dependent column
    df_yNext = df_y.reshape(-1,1)

    # create a regressor object (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    regressorNext = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=1,
               max_leaf_nodes=len(df_XNext), min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               random_state=None, splitter='best')

    # fit the regressor with X and Y data 
    regressorNext.fit(df_XNext, df_yNext) 
    
    return regressorNext.predict(np.array([x_ref]).reshape(1, 1))[0]

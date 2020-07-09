  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


''' TOWARDS
myRidge = MyRidge()
parameters = {‘lambda’, [0.01, 0.1 ,0,1 ,100,1000]}
myRidgeReg = GridSearchCV(..)
myRidgeReg.fit(X,y)
''' 


class MyRidge():
    
    def __init__(self):
        self.coef_ = None
        
    def fit(self, X, y, lambda=1):

        # training data 
        self.data = X
        self.target = y
        
        # Compute covariance matrices
        XtX = X.T.dot(X)
        Xty = X.T.dot(y)
        m = X.shape[0]        
        
        # Do Ridge regression
        coef = np.linalg.inv(XtX +lambda*np.eye(m)).dot(Xty)

        self.coef_ =coef

    def predict(self,X):
        return np.dot(X, self.coef_) 
  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GeneralMetrics:
    
    def sse(self):
        '''returns sum of squared errors (model vs actual)'''
        squared_errors = (self.y_ - self.predict(self.X_)) ** 2
        self.sq_error_ = np.sum(squared_errors)
        return self.sq_error_
        
    def sst(self):
        '''returns total sum of squared errors (actual vs avg(actual))'''
        avg_y = np.mean(self.y_)
        squared_errors = (self.y_ - avg_y) ** 2
        self.sst_ = np.sum(squared_errors)
        return self.sst_
    
    def r_squared(self):
        '''returns calculated value of r^2'''
        self.r_sq_ = 1 - self.sse()/self.sst()
        if self.r_sq_ < 0:
            self.r_sq_ = 0
        return self.r_sq_
    
    def adj_r_squared(self):
        '''returns calculated value of adjusted r^2'''
        self.adj_r_sq_ = 1 - (self.sse()/self._dfe) / (self.sst()/self._dft)
        return self.adj_r_sq_
       
    def mse(self):
        '''returns calculated value of mse'''
        self.mse_ = np.mean( (self.predict(self.X_) - self.y_) ** 2 )
        return self.mse_

    def pretty_print_stats(self):
        '''returns report of statistics for a given model object'''
        items = ( ('sse:', self.sse()), ('sst:', self.sst()), 
                 ('mse:', self.mse()), ('r^2:', self.r_squared()))
        for item in items:
            print('{0:8} {1:.4f}'.format(item[0], item[1]))

    

class RidgeRegression(GeneralMetrics):
    
    def __init__(self):
        self.coef_ = None
        
    def fit(self, X, y,kappa=1):
    
        # training data 
        self.X_ = X
        self.y_ = y
        
        # Compute covariance matrices
        XtX = X.T.dot(X)
        Xty = X.T.dot(y)
        m = X.shape[1]        
        
        # Do Ridge regression
        coef = np.linalg.inv(XtX + kappa*np.eye(m)).dot(Xty)

        self.coef_ =coef

    def predict(self,X):
        return np.dot(X, self.coef_) 

    def resids(self):
        return self.y_ - np.dot(self.X_, self.coef_)
        


# Make some fake data for testing
X = np.random.rand(10,5)
y = np.random.rand(10,1)
kappa=1
myRidge = RidgeRegression()
myRidge.fit(X,y,kappa)

# Print some nice regression statistics
myRidge.pretty_print_stats()

''' NOW CODE TOWARDS META OPTIMIZATION
parameters = {‘lambda’, [0.01, 0.1 ,0,1 ,100,1000]}
myRidgeReg = GridSearchCV(..)
myRidgeReg.fit(X,y)
''' 
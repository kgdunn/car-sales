  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MyRidge():
    
    def __init__(self):
        self.coef_ = None
        
    def fit(self, X, y, Xt , yt, kappa=1):

        # training data & ground truth data
        self.data = X
        self.target = y
        
        # degrees of freedom population dep. variable variance 
        self._dft = X.shape[0] - 1  
            
        # Covariance kernels
        XtX = X.T.dot(X)
        XtX2 = Xt.T.dot(Xt)
        Xty = X.T.dot(y)
        Xty2 = Xt.T.dot(yt)
        
        # Model Transfer
        coef = np.linalg.inv(kappa*XtX +(1-kappa)*XtX2).dot(kappa*Xty+(1-kappa)*Xty2)
        self.coef_ =coef

    def predict(self,X):
        return np.dot(X, self.coef_) 
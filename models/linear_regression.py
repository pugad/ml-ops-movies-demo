from typing import Any
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas
from .basemodel import BaseMLModel
import uuid
import datetime
import numpy as np

class LinearRegressionModel(BaseMLModel):
    '''
    An abstraction of linear_model.LinearRegression
    params:
        -   model_id:           ID of the model. Contains the prefix "LR" as shorthand for the model's name
        -   clf:                (optional) load an already-instantiated model
    '''
    def __init__(
        self,
        model_id:str='LR-' + str(uuid.uuid4())[:8] + '-{}'.format(datetime.datetime.now().date()),
        clf:linear_model.LinearRegression=None
    ):
        # set the id
        self.model_id = model_id

        # initialize the model
        self.clf = clf if clf is not None else linear_model.LinearRegression()

        self.eval_results = {
            'R-squared':None,
            'RMSE':None
        }
    
    def train(self, X_train:pandas.DataFrame, y_train:list) -> linear_model.LinearRegression:
        '''
        Train the Decision linear_model Classifier
        params:
            -   X_train:    features as a pandas DataFrame (or numpy array) containing numerical data (categorical data should already be one-hot encoded)
            -   y_train:    labels/targets as a list containing the expected results (to train our model with).
        returns:
            -   trained linear_model.LinearRegression
        '''
        # train model
        clf_trained = self.clf.fit(X=X_train, y=y_train)
        
        # update model
        self.clf = clf_trained
        
        # return the trained model
        return self.clf
    
    def predict(self, X_test:pandas.DataFrame) -> Any:
        '''
        Predict using the trained model
        params:
            -   X_test: features as pandas DataFrame (or numpy array) containing data to predict with
        returns:
            -   predicted value with the same data type as the labels data used for training
        '''
        
        return self.clf.predict(X_test)
        
    def evaluate(self, X_test:pandas.DataFrame, y_test:list, *args, **kwargs):
        '''
        Evaluate the trained model
        params:
            -   X_test: features as pandas DataFrame (or numpy array) containing data to predict with
            -   y_test: labels as a list

        '''
        # predict using the hold-out test dataset
        y_pred = self.predict(X_test=X_test)

        # compute for R-squared
        clf_r2 = self.clf.score(X_test, y_test)

        # compute RMSE
        clf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # update eval results
        self.eval_results = {
            'R-squared':clf_r2,
            'RMSE':clf_rmse
        }

        return dict(self.eval_results)
import pandas
from models.basemodel import BaseMLModel

def predict_movie_revenue(trained_ml_model:BaseMLModel, inputs:pandas.DataFrame) -> float:
    '''
    Returns a prediction based on the trained model and the provided feature inputs
    
    params:
        -   trained_ml_model: a trained instance of a model
        -   inputs: pandas DataFrame containing the features with the same variable names as the training/test datasets
    
    returns:
        -   float of the predicted gross revenue in USD
    '''
    return trained_ml_model.predict(X_test=inputs)
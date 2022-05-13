from typing import Any


class BaseMLModel:
    '''
    Base ML Model class
    params:
        -   model_id:   ID of the model once instantiated
    '''
    def __init__(self, model_id:str, *args, **kwargs):
        self.model_id = model_id

    def train(self, X_train, y_train, *args, **kwargs):
        pass

    def predict(self, X_test, *args, **kwargs) -> Any:
        pass
    
    def evaluate(self, X_test, y_test, *args, **kwargs) -> dict:
        pass
from models.basemodel import BaseMLModel


class BaseRegistry:

    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def persist(self, model:BaseMLModel, *args, **kwargs):
        '''
        Persists a trained model into the registry.
        '''
        pass
    
    def load(self, model_id:str, *args, **kwargs):
        '''
        Loads and instantiates a model from the registry.
        '''
        pass
    
    def model_exists(self, model_id:str, *args, **kwargs):
        '''
        Check if a model with the given model_id already exists in the registry.
        '''
        pass
    
    def save_eval_results(self, model_id:str, eval_results:dict, *args, **kwargs):
        '''
        Save a model's R-squared and RMSE scores
        '''
        pass
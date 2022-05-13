from models.basemodel import BaseMLModel
from models.decision_tree_regressor import DecisionTreeRegressorModel
from models.linear_regression import LinearRegressionModel
from models.random_forest_regressor import RandomForestRegressorModel

from model_registry.base_registry import BaseRegistry

import joblib
import os
import json
import pandas as pd
import datetime

class LocalRegistry(BaseRegistry):

    def __init__(self, path:str) -> None:
        self.base_path = path
        self.model_classes = {
            'DTR':DecisionTreeRegressorModel,
            'LR':LinearRegressionModel,
            'RFR':RandomForestRegressorModel
        }

        # create a models folder to persist all the trained models
        # regardless if they've passed evaluation
        self.models_path = os.path.join(self.base_path,'models')
        if not os.path.exists(self.models_path):
            os.mkdir(self.models_path)

        # create a folder for model metadata
        # e.g. to store evaluation/benchmark scores
        self.metadata_path = os.path.join(self.base_path,'metadata')
        if not os.path.exists(self.metadata_path):
            os.mkdir(self.metadata_path)
        
        # create a metadata file if one does not exist yet
        metadata_filename = 'metadata.json'
        self.metadata_filepath = os.path.join(self.metadata_path, metadata_filename)
        if not os.path.exists(self.metadata_filepath):
            with open(self.metadata_filepath, 'w+') as f:
                f.write(json.dumps([]))

    def persist(self, model:BaseMLModel):
        '''
        Persists the ML model to the local directory/filepath using joblib.
        '''

        joblib.dump(model, os.path.join(self.models_path, model.model_id + '.joblib'))
    
    def load(self, model_id:str):
        '''
        Loads and instantiates a model from the local directory using joblib.load
        '''

        # Identify the model's class
        model_class = model_id.split('-')[0]

        model_filepath = os.path.join(self.models_path, model_id+'.joblib')

        return self.model_classes[model_class](model_id=model_id, clf=joblib.load(model_filepath))
    
    def model_exists(self, model_id:str):
        '''
        Check if a model with the given model_id already exists in the datastore.
        '''
        filename = model_id + '.joblib'
        return os.path.exists(os.path.join(self.models_path, filename))
    
    def save_eval_results(self, model_id:str, eval_results:dict):
        '''
        Save a model's R-squared and RMSE scores as a json file.
        '''
        metrics = {
            'model_id':model_id,
            'R-squared':eval_results['R-squared'],
            'RMSE':eval_results['RMSE'],
            'timestamp':str(datetime.datetime.now())
        }

        with open(self.metadata_filepath, 'r') as f:
            data = json.loads(f.read())
        
        data.insert(0, metrics)

        with open(self.metadata_filepath, 'w+') as f:
            f.write(json.dumps(data))


    
    def load_eval_results(self, model_id:str):
        '''
        Loads a model's R-squared and RMSE scores from a json file.
        '''

        with open(self.metadata_filepath, 'r') as f:
            data = json.loads(f.read())
        
        r = [result for result in data if result['model_id'] == model_id]
        if not r:
            return None
        return r[0]
    
    def get_best_model_by_model_id(self, benchmark:dict) -> str:

        with open(self.metadata_filepath, 'r') as f:
            data = json.loads(f.read())
        
        evaldf = pd.DataFrame(data)

        # filter to get the models that are within the expected benchmark scores
        result = evaldf[(evaldf['R-squared'] >= benchmark['R-squared']) & (evaldf['RMSE'] <= benchmark['RMSE'])].sort_values(by=['R-squared','RMSE'], ascending=False)['model_id'].values

        if len(result) > 0:
            return result[0]
        else:
            return ''
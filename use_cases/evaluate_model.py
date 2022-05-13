import pandas as pd
from models.basemodel import BaseMLModel

def model_passes_eval(model_eval_results:dict, benchmark:dict):
    '''
    Checks whether a model's evaluation results are within or better
    than expected R-squared and RMSE scores.

    params:
        -   model_eval_results:         dictionary containing a model's "R-squared" and "RMSE" scores
        -   benchmark:                  dictionary containing the standard/benchmark "R-squared" and "RMSE" scores
    '''

    # check if the model's scores are equal to or better than the benchmark scores.
    # If they are, return True.
    if (
        model_eval_results['R-squared'] >= benchmark['R-squared'] and
        model_eval_results['RMSE'] <= benchmark['RMSE']
    ):
        return True
    
    return False

def get_best_model(list_of_models:list):
    '''
    Compares the .eval_results scores of each model and returns
    the model with the best scores, with a priority on highest R-squared.
    '''
    models = pd.DataFrame([{'model':m, 'R-squared':m.eval_results['R-squared'], 'RMSE':m.eval_results['RMSE']} for m in list_of_models])
    return models.sort_values(by=['R-squared', 'RMSE'], ascending=False)['model'].values[0]
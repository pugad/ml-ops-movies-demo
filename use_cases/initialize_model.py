from models.basemodel import BaseMLModel

from models.decision_tree_regressor import DecisionTreeRegressorModel
from models.linear_regression import LinearRegressionModel
from models.random_forest_regressor import RandomForestRegressorModel


def create_train_persist_model(
    model_id:str,
    registry_interface,
    X_train,
    y_train,
    X_test,
    y_test,
    random_state:int=None,
    persist_model:bool=False
) -> BaseMLModel:

    # create a new model
    new_model = create_model(model_id=model_id, random_state=random_state)

    # train the new model
    new_model.train(X_train=X_train, y_train=y_train)

    # evaluate the newly-trained model
    eval_results = new_model.evaluate(X_test=X_test, y_test=y_test)

    # persist the model
    if persist_model:
        registry_interface.persist(new_model)
    
    # save the evaluation results
    registry_interface.save_eval_results(model_id=model_id, eval_results=eval_results)

    return new_model



def initialize_model(
    model_id:str,
    registry_interface,
    random_state:int=None,
    persist_model=False
) -> BaseMLModel:
    '''
    Checks if a model of the given ID already exists in the model registry
    and loads that model. Otherwise, a new model is created and persisted in the registry.
    params:
        -   model_id:               ID of the model to be initialized. Must contain a prefix ("DTR","LR" or "RFR")
                                    to indicate the model's class.
        -   registry_interface:     Interface class that manages model persistence and loading.
        -   random_state:           Applicable only for the Decision Tree and Random Forest Regressors.
                                    Takes an integer that makes Decision Tree / Random Forest Regressors deterministic.
        -   persist_model:          Set to True if a newly-created model should be persisted in the registry.
    '''

    # check if the model already exists in the registry
    if registry_interface.model_exists(model_id=model_id):
        return registry_interface.load(model_id=model_id)
    
    # create a new model otherwise
    new_model = create_model(model_id=model_id, random_state=random_state)

    # persist the new model if specified
    if persist_model:
        registry_interface.persist(model=new_model)
    
    return new_model



def create_model(model_id:str, random_state:int=None) -> BaseMLModel:
    '''
    Instantiates a new model based on the model ID.

    Params:
        -   model_id:       ID of the model to be instantiated. Must contain a prefix ("DTR","LR" or "RFR")
                            to indicate the model's class.
        -   random_state:   Applicable only for the Decision Tree and Random Forest Regressors.
                            Takes an integer that makes Decision Tree / Random Forest Regressors deterministic.
    '''

    model_classes = {
        'LR':LinearRegressionModel,
        'DTR':DecisionTreeRegressorModel,
        'RFR':RandomForestRegressorModel
    }

    # get the shorthand name for the model class
    prefix = model_id.split('-')[0]

    if prefix == 'LR':
        return model_classes[prefix](model_id=model_id)
    else:
        return model_classes[prefix](model_id=model_id, random_state=random_state)
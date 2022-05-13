from models.basemodel import BaseMLModel
from models.decision_tree_regressor import DecisionTreeRegressorModel
from models.linear_regression import LinearRegressionModel
from models.random_forest_regressor import RandomForestRegressorModel

# TODO: add more specific tests

def test_base_model():
    assert 'train' in dir(BaseMLModel)
    assert 'predict' in dir(BaseMLModel)
    assert 'evaluate' in dir(BaseMLModel)

def test_decision_tree_regressor():
    assert 'train' in dir(DecisionTreeRegressorModel)
    assert 'predict' in dir(DecisionTreeRegressorModel)
    assert 'evaluate' in dir(DecisionTreeRegressorModel)

def test_linear_regression_model():
    assert 'train' in dir(LinearRegressionModel)
    assert 'predict' in dir(LinearRegressionModel)
    assert 'evaluate' in dir(LinearRegressionModel)

def test_random_forest_regressor():
    assert 'train' in dir(RandomForestRegressorModel)
    assert 'predict' in dir(RandomForestRegressorModel)
    assert 'evaluate' in dir(RandomForestRegressorModel)
from models.basemodel import BaseMLModel

def load_model(model_id:BaseMLModel, registry_interface):
    '''
    Loads and instantiates a model that was persisted in a registry.
    '''
    return registry_interface.load(model_id=model_id)
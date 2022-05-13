from models.basemodel import BaseMLModel

def persist_model(model:BaseMLModel, registry_interface, *args, **kwargs):
    '''
    Persists an ML model in a registry.
    params:
        -   model:                  Instantiated model/classifier
        -   registry_interface:     registry interface class that implements the persistent storage of the model
    '''
    return registry_interface.persist(model=model, *args, **kwargs)



from models.datamodels import PredictDataModel

from use_cases.initialize_model import create_train_persist_model
from use_cases.load_model import load_model
from use_cases.predict_movie_revenue import predict_movie_revenue

from repositories.githubrepo import get_movies_dataset

from model_registry.local_registry import LocalRegistry


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


import os
import uuid
import datetime

# Initialize FastAPI.
APP = FastAPI()

# Mount static and templates folders.
APP.mount('/static', StaticFiles(directory='static'), name='static')
TEMPLATES = Jinja2Templates(directory='templates')

# Get the URL to the dataset
MOVIES_DATASET_URL = os.environ.get('DATASET_SOURCE_URL', None)

# For this simple example, we'll make the models persist in a local folder
# (as implemented in model_registry/local_registry.py).
# In production, we can write another registry interface to a cloud server
# that will manage the persisted models.

PERSISTENCE_DIR = 'persist_dir'
if not os.path.exists(PERSISTENCE_DIR):
    os.mkdir(PERSISTENCE_DIR)

# Initialize model registry
MODEL_REGISTRY = LocalRegistry(path=PERSISTENCE_DIR)

# Initialize model baseline scores/benchmarks
MODEL_BENCHMARKS = {
    'R-squared':float(os.environ['MODEL_BASELINE_R2']),
    'RMSE':int(os.environ['MODEL_BASELINE_RMSE'])
}

## Initialize model IDs.
# If there's no specific model ID specified in the environment variables, generate a new one.

# Get or generate Linear Regression model's current ID
MODEL_ID_LR = os.environ.get('MODEL_ID_LR','LR-' + str(uuid.uuid4())[:8] + '-' + str(datetime.datetime.now().date()))

# Get or generate Decision Tree Regressor's current ID
MODEL_ID_DTR = os.environ.get('MODEL_ID_DTR','DTR-' + str(uuid.uuid4())[:8] + '-' + str(datetime.datetime.now().date()))

# Get or generate Random Forest Regressor's current ID
MODEL_ID_RFR = os.environ.get('MODEL_ID_RFR','RFR-' + str(uuid.uuid4())[:8] + '-' + str(datetime.datetime.now().date()))

# Get the features and labels
DATASETS = get_movies_dataset(MOVIES_DATASET_URL)

def run_pipeline():
    '''
    Run the pipeline:
    Instantiate, train, evaluate, and persist the models.
    '''

    # If the model doesn't yet exist in the registry,
    # instantiate, train, evaluate, and persist the models.


    # Linear Regression Model
    if not MODEL_REGISTRY.model_exists(model_id=MODEL_ID_LR):
        create_train_persist_model(
            model_id=MODEL_ID_LR,
            registry_interface=MODEL_REGISTRY,
            X_train=DATASETS['X_train'],
            y_train=DATASETS['y_train'],
            X_test=DATASETS['X_test'],
            y_test=DATASETS['y_test'],
            random_state=None,
            persist_model=True
        )
    
    # Decision Tree Regressor
    if not MODEL_REGISTRY.model_exists(model_id=MODEL_ID_DTR):
        create_train_persist_model(
            model_id=MODEL_ID_DTR,
            registry_interface=MODEL_REGISTRY,
            X_train=DATASETS['X_train'],
            y_train=DATASETS['y_train'],
            X_test=DATASETS['X_test'],
            y_test=DATASETS['y_test'],
            random_state=int(os.environ.get('MODEL_RANDOM_STATE',0)),
            persist_model=True
        )
    
    # Random Forest Regressor
    if not MODEL_REGISTRY.model_exists(model_id=MODEL_ID_RFR):
        create_train_persist_model(
            model_id=MODEL_ID_RFR,
            registry_interface=MODEL_REGISTRY,
            X_train=DATASETS['X_train'],
            y_train=DATASETS['y_train'],
            X_test=DATASETS['X_test'],
            y_test=DATASETS['y_test'],
            random_state=int(os.environ.get('MODEL_RANDOM_STATE',0)),
            persist_model=True
        )

# initial run of the pipeline on app start
run_pipeline()


@APP.get("/")
async def home(request:Request):
    return TEMPLATES.TemplateResponse("index.html", context={'request':request, 'dataset_source_url':MOVIES_DATASET_URL}, status_code=200)


@APP.get("/api/docs")
async def docs(request:Request):
    return TEMPLATES.TemplateResponse("docs.html", context={'request':request}, status_code=200)

@APP.get("/healthz")
async def healthz(request:Request):
    '''
    Healthz endpoint to tell a K8s cluster whether or not our API is ready
    to be served/exposed.
    '''
    # For this demo, we'll just check if the movies dataset URL exists.
    # In production, we can ping the persistent storage provider here.
    # If it fails, we return a non-2XX status code, such as 503
    if not MOVIES_DATASET_URL:
        return JSONResponse(content={
            'status':'not ready',
            'details':'movies dataset URL not found / unavailable'
        }, status_code=503)
    
    return JSONResponse(content={'status':'ok'}, status_code=200)

@APP.post("/train")
async def train():
    '''
    This is a trigger to re-run the pipeline (re-train the models).
    '''

    # Get the latest copy of the features and labels
    DATASETS = get_movies_dataset(MOVIES_DATASET_URL)

    # Generate a new set of IDs for each of the models

    # Linear Regression
    MODEL_ID_LR = 'LR-' + str(uuid.uuid4())[:8] + '-' + str(datetime.datetime.now().date())

    # Decision Tree Regressor
    MODEL_ID_DTR = 'DTR-' + str(uuid.uuid4())[:8] + '-' + str(datetime.datetime.now().date())

    # Random Forest Regressor
    MODEL_ID_RFR = 'RFR-' + str(uuid.uuid4())[:8] + '-' + str(datetime.datetime.now().date())

    run_pipeline()

    return JSONResponse(content={'status':'ok', 'details':'models trained'}, status_code=200)

@APP.post("/predict")
async def predict(data:PredictDataModel):
    '''
    Performs a prediction using the trained and persisted model.
    '''

    # Validate input data.
    predict_df = data.to_dataframe()
    if not len(predict_df):
        return JSONResponse(
            content={'status':'error','reason':'invalid/empty data'},
            status_code=400
        )

    # Get the best model's model_id
    model_id_best = MODEL_REGISTRY.get_best_model_by_model_id(benchmark=MODEL_BENCHMARKS)

    # if no models passed the evaluation / benchmark, return 503 Service Unavailable
    if not model_id_best:
        return JSONResponse(
            content={
                'status':'not ready',
                'details':'models not available to be served'
            },
            status_code=503
        )

    # Load the best model. This is to ensure
    # that we're always using the latest best model, in case
    # another running pipeline instance updates the models.
    model_best = load_model(model_id=model_id_best, registry_interface=MODEL_REGISTRY)

    # Run the prediction.
    predicted_label = predict_movie_revenue(model_best, predict_df)

    return JSONResponse(content={
        'status':'ok',
        'prediction':predicted_label[0]
    }, status_code=200)


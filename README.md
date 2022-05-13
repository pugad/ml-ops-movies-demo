# Predict a movie's revenue!
This project incorporates both the ML pipeline and prediction service to estimate how much revenue a movie will make.

## Try it out

### Set up

Prerequisites:

- Docker
- Python 3.9

Clone the repo

    git clone https://github.com/pugad/ml-ops-movies-demo 
    cd ml-ops-movies-demo 

Spin up the container

    docker-compose up -d --build

Note: if you want to see the persisted models and metadata, uncomment the line "./:/usr/src/app" from the docker-compose.yaml file. A "persist_dir" folder should show up in your project directory after you spin up the container.

### Check out the docs

Public API docs: http://localhost:8000/api/docs


FastAPI Swagger UI Docs: http://localhost:8000/docs

### Interact with the API

Using the UI at the homepage: http://localhost:8000/

Using the FastAPI Swagger UI Docs: http://localhost:8000/docs

Using Python:

    # Create a virtual environment and install requests
    python -m venv venv
    .\venv\Scripts\activate
    pip install -U requests

    # You can edit the payload at manualtests/manual_predict_data.json
    code .\manualtests\

    # Send a prediction request to /predict
    python .\manualtests\predict_manual.py localhost:8000/predict

# About
The pipeline covers the data preparation (data extraction, cleaning, feature engineering), training (model training, evaluation, and persistence), and inference phases (choosing the best model to serve predictions). FastAPI is used to build the APIs. Three models are trained and compared: Linear regression, Decision tree regressor, and the Random forest regressor. Then, the model with the best R-squared and RMSE scores is served.

The baseline scores were calculated during development (see *references/development/compute_best_comb.py*):
- Baseline R-squared: 0.668
- Baseline Root Mean Square Error (RMSE): 43,100,000

The original development notebook can be found in *references/original_notebook_ref.ipynb*

Credits to Thinkful for the original box office dataset.

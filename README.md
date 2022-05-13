# ml-ops-movies-demo
This demo MLOps project uses machine learning to predict a movie's revenue!

The pipeline covers the data preparation (data extraction, cleaning, feature engineering), training (model training, evaluation, and persistence), and inference phases (choosing and serving models for predictions). It trains, evaluates, and compares three models (linear regression, decision tree, random forest) and then serves the model with the best R-squared and RMSE scores based on a baseline standard (see .configs file).

The baseline scores were calculated during development (see references/development/compute_best_comb.py):
R-squared must be greater than or equal to 0.668
Root Mean Square Error (RMSE) must be less than or equal to 43,100,000

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


### Check out the docs

Public API docs: http://localhost:8000/api/docs


FastAPI Swagger UI Docs: http://localhost:8000/docs

### Interact with the API

Manually at the homepage: http://localhost:8000/

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

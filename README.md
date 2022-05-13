# ml-ops-movies-demo
This demo MLOps project uses machine learning to predict the revenue of a movie!

The pipeline covers the data preparation (data extraction, cleaning, feature engineering), training (model training, evaluation, and persistence), and inference phases (prediction service). It trains and evaluates three models (linear regression, decision tree, random forest) and then serves the model with the best R-squared and RMSE scores based on a set standard.

Based on the scores obtained during development, the current standard is set as follows:
R-squared must be greater than or equal to 66.8%
Root Mean Square Error must be less than or equal to 43.1M

## Try it out

### Set up

Clone the repo

    git clone https://github.com/pugad/ml-ops-movies-demo 
    cd ml-ops-movies-demo 

Spin up the container

    docker-compose up -d --build

Create a virtual environment and install requests

    python -m venv venv
    .\venv\Scripts\activate
    pip install -U requests

### Interact with the API

Open the homepage

    http://localhost:8000/
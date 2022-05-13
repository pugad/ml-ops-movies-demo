# data analysis packages
import numpy as np
import pandas as pd

# plotting package
import matplotlib.pyplot as plt

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import itertools
from tqdm import tqdm

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

def clean_data(subset):

    # import data from github
    data = pd.read_csv("https://raw.githubusercontent.com/pugad/predict_the_box_office/master/movie_metadata.csv")
    # DATA CLEANING
    # Rename columns to include units
    data.rename(columns={'duration': 'duration_mins',
                        'budget': 'budget_usd',
                        'gross': 'gross_usd'}, inplace=True)

    # Drop all duplicate movie titles that were released in the same year
    data = data.drop_duplicates(subset=['movie_title', 'title_year'], keep='first').copy()

    # Drop the aspect ratio column,  axis=1 means drop the column
    data.drop('aspect_ratio', axis=1, inplace=True)

    # Drop all null values of gross
    # You will be predicting this value
    # so don't want to impute these values (skew the analysis)
    data.dropna(subset=['gross_usd'], how='all', inplace=True)

    # Drop movies where the year is missing
    data.dropna(subset=['title_year'], how='all', inplace=True)
    # Convert all years to integers
    data['title_year'] = data['title_year'].astype(int)

    # Calculate median budgets per year
    # Impute the median budgets per year for missing budget data
    data['budget_usd'] = data['budget_usd'].fillna(data.groupby('title_year')['budget_usd'].transform('median'))

    # Drop the remaining row that is missing budget
    # (no other movies from 1942 in the dataset)
    data.dropna(subset=['budget_usd'], axis=0, inplace=True)

    # Find how many movies are in each country in the data
    counts = data['country'].value_counts()
    # Select the data from only the top 3 countries
    data = data[data['country'].isin(counts.nlargest(3).index)].copy()

    # Dropping all remaining rows that have null values
    data.dropna(axis=0, inplace=True)

    # FEATURE ENGINEERING
    # Identify all movie counts, select all star actors
    lead_movie_counts = data['actor_1_name'].value_counts()
    star_actors = lead_movie_counts[lead_movie_counts >= 20].index
    # Set `lead_star` = 1 if actor is in star_actors, otherwise 0
    data['lead_star'] = [1 if x in star_actors else 0 for x in data['actor_1_name']]

    data = data[subset + ['gross_usd']]

    if 'content_rating' in subset:
        # Encoding ratings as dummy variables
        content_ratings = pd.get_dummies(data['content_rating'])
        # Merge the encoded data back on to the original data
        data = data.join(content_ratings)

    # Select columns by data type - number
    numerical_data = data.select_dtypes(include='number')
    # numerical_data.head()
    numerical_data
    
    # Identifying our Feature set (X) and target (y) variables for modeling
    X = numerical_data.drop(['gross_usd'], axis=1)
    y = numerical_data['gross_usd']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return {
        'X_train':X_train,
        'X_test':X_test,
        'y_train':y_train,
        'y_test':y_test
    }

def compute_linear_reg(datasets):
    X_train = datasets['X_train']
    X_test = datasets['X_test']
    y_train = datasets['y_train']
    y_test = datasets['y_test']

    # Step 1: Instantiating the model
    lr = LinearRegression()

    # Step 2: Fit the model (note, we use the train set here)
    lr.fit(X_train, y_train)

    # Step 3: Make predictions (note, we use the test set here)
    y_pred_lr = lr.predict(X_test)

    # Step 4: Evaluate the model (note, we use the test set here)
    lr_r2 = lr.score(X_test, y_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    
    return {
        'R2':lr_r2,
        'RMSE':lr_rmse
    }

def compute_decision_tree(datasets):
    X_train = datasets['X_train']
    X_test = datasets['X_test']
    y_train = datasets['y_train']
    y_test = datasets['y_test']

    # Step 1: Instantiating the model
    dt = DecisionTreeRegressor(random_state=42)

    # Step 2: Fit the model (note, we use the train set here)
    dt.fit(X_train, y_train)

    # Step 3: Make predictions (note, we use the test set here)
    y_pred_dt = dt.predict(X_test)

    # Step 4: Evaluate the model (note, we use the test set here)

    dt_r2 = dt.score(X_test, y_test)

    dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))


    return {
        'R2':dt_r2,
        'RMSE':dt_rmse
    }


def compute_random_forest(datasets):
    X_train = datasets['X_train']
    X_test = datasets['X_test']
    y_train = datasets['y_train']
    y_test = datasets['y_test']

    # Step 1: Instantiating the model
    rfr = RandomForestRegressor()

    # Step 2: Fit the model (note, we use the train set here)
    rfr.fit(X_train, y_train)

    # Step 3: Make predictions (note, we use the test set here)
    y_pred_rf = rfr.predict(X_test)

    # Step 4: Evaluate the model (note, we use the test set here)

    rf_r2 = rfr.score(X_test, y_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))


    return {
        'R2':rf_r2,
        'RMSE':rf_rmse
    }


def compute_models(subset):
    datasets = clean_data(subset)
    
    linreg = compute_linear_reg(datasets)
    dectree = compute_decision_tree(datasets)
    ranfor = compute_random_forest(datasets)

    return {
        'model_name':['Linear Regression', 'Decision Tree','Random Forest'],
        'R2':[linreg['R2'], dectree['R2'], ranfor['R2']],
        'RMSE':[linreg['RMSE'], dectree['RMSE'], ranfor['RMSE']],
        'combination':[subset]*3
    }

def main():

    # except for num_voted_users, all the features below
    # are information that can be determined prior to a movie's release 
    fullset = [
        'num_voted_users',
        'budget_usd',
        'content_rating',
        'title_year',
        'duration_mins',
        'actor_3_facebook_likes',
        'cast_total_facebook_likes',
        'director_facebook_likes',
        'facenumber_in_poster',
        'lead_star',
        'actor_1_facebook_likes',
        'actor_2_facebook_likes',
        'movie_facebook_likes',
    ]

    # get all possible combinations of 10 features out of the 13 total
    combinations = list(itertools.combinations(fullset, 10))

    # initialize the resulting dictionary
    results_dict = {
        'model_name':[],
        'R2':[],
        'RMSE':[],
        'combination':[],
    }
    cols = list(results_dict)


    with ThreadPoolExecutor() as executor:
        future_to_comb = [executor.submit(compute_models, list(comb)) for comb in combinations]

        with tqdm(total=len(future_to_comb)) as pbar:
            for future in concurrent.futures.as_completed(future_to_comb):
                
                call_result = future.result()
                [results_dict[key].extend(call_result[key]) for key in cols]
                
                pbar.update(1)
        
    
    results_df = pd.DataFrame(results_dict)
    results_df.to_excel('R2_RMSE_MODEL_ANALYSIS.xlsx', index=False)


if __name__ == '__main__':

    main()




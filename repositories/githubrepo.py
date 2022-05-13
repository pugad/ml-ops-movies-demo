import pandas as pd
from sklearn.model_selection import train_test_split


def get_movies_dataset(url) -> dict:
    '''
    The job of this function is to get the latest
    version of the dataset from a URL. Since we're directly obtaining
    the data, we'll need to do some cleaning and feature engineering.
    
    If in case we decide to change to a feature store that exposes an API for
    obtaining specific versions of the dataset, then we can update this function
    to interact with that API accordingly.
    '''

    # Import using pandas.
    # If the file is too large, then we can refactor this line to use dask instead
    # (see helpers.csv_loaders.py)
    df = pd.read_csv(url)

    # clean dataset
    df = clean_movies_dataset(df)
    
    # engineer features
    df = engineer_features_movies_dataset(df)

    # prepare feature matrix and labels (we are trying to predict the gross revenue of movies)
    X = df.drop(['gross_usd'], axis=1)
    y = df['gross_usd']

    # split training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # return the training and test sets
    return {
        'X_train':X_train,
        'X_test':X_test,
        'y_train':y_train,
        'y_test':y_test
    }

def clean_movies_dataset(data):
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

    # convert likes to integers
    data['actor_3_facebook_likes'] = data['actor_3_facebook_likes'].astype(int)
    data['actor_2_facebook_likes'] = data['actor_2_facebook_likes'].astype(int)
    data['actor_1_facebook_likes'] = data['actor_1_facebook_likes'].astype(int)
    data['cast_total_facebook_likes'] = data['cast_total_facebook_likes'].astype(int)
    data['director_facebook_likes'] = data['director_facebook_likes'].astype(int)

    

    return data

def engineer_features_movies_dataset(data):
    # FEATURE ENGINEERING
    # Identify all movie counts, select all star actors
    lead_movie_counts = data['actor_1_name'].value_counts()
    star_actors = lead_movie_counts[lead_movie_counts >= 20].index
    # Set `lead_star` = 1 if actor is in star_actors, otherwise 0
    data['lead_star'] = [1 if x in star_actors else 0 for x in data['actor_1_name']]

    # Let's analyze with just the select subset of features
    data = data[['num_voted_users',
        'budget_usd',
        'content_rating',
        'title_year',
        'duration_mins',
        'actor_3_facebook_likes',
        'cast_total_facebook_likes',
        'director_facebook_likes',
        'actor_1_facebook_likes',
        'actor_2_facebook_likes',
        'gross_usd'
    ]]

    # Encoding ratings as dummy variables
    content_ratings = pd.get_dummies(data['content_rating'])
    # Merge the encoded data back on to the original data
    data = data.join(content_ratings)

    # Select columns by data type - number
    numerical_data = data.select_dtypes(include='number')

    # numerical_data.head()
    return numerical_data
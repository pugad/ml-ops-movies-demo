from pydantic import BaseModel
from typing import Literal
import pandas as pd


class PredictDataModel(BaseModel):
    '''
    Data model to validate a POST payload to /predict
    '''
    num_voted_users: int
    budget_usd: float
    content_rating: Literal[
            'Approved',
            'G',
            'GP',
            'M',
            'NC-17', 
            'Not Rated', 
            'PG',
            'PG-13',
            'Passed',
            'R',
            'Unrated',
            'X'
        ]
    title_year: int
    duration_mins: float
    actor_3_facebook_likes: int
    cast_total_facebook_likes: int
    director_facebook_likes: int
    actor_1_facebook_likes: int
    actor_2_facebook_likes: int

    def clean(self) -> dict:
        '''
        Cleans the received payload.
        '''
        content_ratings = [
            'Approved',
            'G',
            'GP',
            'M',
            'NC-17', 
            'Not Rated', 
            'PG',
            'PG-13',
            'Passed',
            'R',
            'Unrated',
            'X'
        ]

        data = self.dict()
        rating = data.pop('content_rating')
        [data.update({k:[data[k]]}) for k in data]
        [data.update({r:[1 if rating == r else 0]}) for r in content_ratings]

        return data
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.clean())
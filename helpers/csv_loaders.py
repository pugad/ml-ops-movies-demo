import dask
from dask import dataframe as dd
import csv

def load_large_csv_from_zip(zipped_folder_path, *args, **kwargs) -> dask.dataframe.core.DataFrame:
    '''
    Create a dask dataframe from an archived .csv dataset
    '''
    return dd.read_csv(zipped_folder_path, compression='zip', *args, **kwargs)

def generator_from_csv(filepath, delimiter=','):
    '''
    Returns a generator to iterate over a .csv file.
    Useful for large datasets that are stored locally.
    '''
    with open(filepath, newline='') as f:
        for row in csv.reader(f, delimiter=delimiter):
            yield row

def load_large_csv_from_url(urlpath, dtypes={}) -> dask.dataframe.core.DataFrame:
    return dd.read_csv(urlpath, dtypes=dtypes)
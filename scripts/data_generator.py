import pandas as pd
import numpy as  np
import logging

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/data_generator.log"),
        logging.StreamHandler()
    ]
)

def load_data(train_path, test_path, store_path):
    """
    Load the train, test, and store datasets with specified parameters.
    
    Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the test dataset.
        store_path (str): Path to the store dataset.
        
    Returns:
        tuple: DataFrames for train, test, and store datasets.
    """
    try:
        train = pd.read_csv(train_path, parse_dates=True, low_memory=False)
        test = pd.read_csv(test_path, parse_dates=True, low_memory=False)
        store = pd.read_csv(store_path, low_memory=False)
        logging.info("Data loaded successfully")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise e
    
    return train, test, store

 # Sort data first by store then by date
def sort_train_data(df):
    df.sort_values(by = ['Store', 'Date'], inplace = True)
    return df 

# Merge the two data on Store Column
def merge_data(df1, df2):
    merge_df = pd.merge(df1, df2, on = 'Store')
    return merge_df

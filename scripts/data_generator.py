import pandas as pd
import numpy as  np

# load data
def load_train_data(filepath, date):
    df = pd.read_csv(filepath, parse_dates= [date], low_memory=False)
    return df

def load_store_data(filepath):
    df = pd.read_csv(filepath)
    return df

 # Sort data first by store then by date
def sort_train_data(df):
    df.sort_values(by = ['Store', 'Date'], inplace = True)
    return df 

# Merge the two data on Store Column
def merge_data(df1, df2):
    merge_df = pd.merge(df1, df2, on = 'Store')
    return merge_df

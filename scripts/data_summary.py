import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler('../logs/data_summary.log')
                    ])


def summary_statistics(df):
    """
    Summarize information about the Dataset
    """
    print("Shape of the Data")
    print(df.shape)

    print("\n\ndata Summary")
    print(df.info())

    print("\n\nDescriptive analysis for numerical Column")
    print(df.describe())

    print("\n\nDescriptive analysis for Object Data")
    print(df.describe(include=['object']))

    print("\n\nCheck for Missing Value")
    print(df.isnull().sum().sort_values(ascending=False))


def check_data_types(df):
    """
    Check the data types of features in the dataframe.
    Args:
        df (pd.DataFrame): The dataframe to check.
    Returns:
        dict: Dictionary containing data types for each feature.
    """
    data_types = df.dtypes.to_dict()
    logging.info("Data types checked")
    return data_types
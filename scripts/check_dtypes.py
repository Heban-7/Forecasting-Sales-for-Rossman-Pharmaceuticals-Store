import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler('../logs/check_dtypes.log')
                    ])

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
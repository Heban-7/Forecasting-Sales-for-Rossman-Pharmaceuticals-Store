import logging
import logging.handlers
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers= [
        logging.FileHandler('../logs/data_quality.log'),
        logging.StreamHandler()
    ]
)

def check_data_quality(df):
    """
    Check the quality of the dataset by inspecting the first few rows,
    data types, and missing values.
    Args:
        df (pd.DataFrame): The dataset to check.
    Returns:
        dict: A dictionary containing data quality information.
    """
    quality_info = {
        "head": df.head(),
        "dtypes": df.dtypes,
        "missing_values": df.isnull().sum()
    }
    logging.info("Data quality check completed")
    return quality_info
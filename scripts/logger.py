import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import logging
import logging
import os

# Difene the paths for logs directory
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'log')

# create log directory based on the log_dir 
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# define the file path 
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

#Create Handler 
info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

# Create formatter and set it to the handlers
formatter = logging.Formatter('%(asctime)s -%(levelname)s -%(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Create a Logger and set its level
logger = logging.getLogger()
# Capture all info 
logger.setLevel(logging.INFO)
logger.addHandler(info_handler)
logger.addHandler(error_handler)

def load_data(file_path):
    logger.info("Loading data from file...")
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        logger.info(f"Data loaded with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def plot_sales_over_time(data, store_id):
    logger.info(f"Plotting sales over time for Store {store_id}...")
    try:
        store_data = data[data['Store'] == store_id]
        plt.figure(figsize=(12, 7))
        plt.plot(store_data.index, store_data['Sales'], marker='o', linestyle='-', color='b')
        plt.title(f'Sales over Time for Store {store_id}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting sales over time for Store {store_id}: {e}")


def plot_sales_over_time_all_store(data):
    logger.info("Plotting sales over time for all stores...")
    try:
        plt.figure(figsize=(12, 7))
        plt.plot(data.index, data['Sales'], marker='o', linestyle='-', color='b')
        plt.title('Sales over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting sales over time: {e}")


def weekly_sales_plot(df):
    logger.info("Plotting weekly sales...")
    try:
        weekly_sales = df['Sales'].resample('W').sum()
        plt.figure(figsize=(15, 7))
        plt.plot(weekly_sales.index, weekly_sales)
        plt.title('Weekly Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting weekly sales: {e}")


def monthly_sales_plot(df):
    logger.info("Plotting monthly sales...")
    try:
        monthly_sales = df['Sales'].resample('M').sum()
        plt.figure(figsize=(15, 7))
        plt.plot(monthly_sales.index, monthly_sales)
        plt.title('Monthly Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting monthly sales: {e}")


def plot_yearly_sales(data):
    logger.info("Plotting yearly sales...")
    try:
        df = data.copy()
        df['Year'] = df.index.year
        df['Month'] = df.index.month

        df_plot = df.groupby(['Month', 'Year'])['Sales'].mean().reset_index()
        years = df_plot['Year'].unique()
        colors = plt.cm.tab10.colors

        plt.figure(figsize=(12, 8))
        for i, y in enumerate(years):
            yearly_data = df_plot[df_plot['Year'] == y]
            plt.plot('Month', 'Sales', data=yearly_data, color=colors[i], label=y)

        plt.title('Seasonal Sales Plot by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Sales')
        plt.legend(title='Year')
        plt.grid(True)
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting yearly sales: {e}")


def seasonal_decomposition(df):
    logger.info("Performing seasonal decomposition...")
    try:
        monthly_sales = df['Sales'].resample('M').sum()
        result = seasonal_decompose(monthly_sales, model='additive')
        result.plot()
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()
    except Exception as e:
        logger.error(f"Error performing seasonal decomposition: {e}")


def plot_acf_pacf(df):
    logger.info("Plotting ACF and PACF...")
    try:
        monthly_sales = df['Sales'].resample('M').sum()
        n_lags = len(monthly_sales) // 3
        acf_values = acf(monthly_sales.dropna(), nlags=n_lags)
        pacf_values = pacf(monthly_sales.dropna(), nlags=n_lags)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        ax1.stem(range(len(acf_values)), acf_values)
        ax2.stem(range(len(pacf_values)), pacf_values)
        ax1.set_title('Autocorrelation Function')
        ax2.set_title('Partial Autocorrelation Function')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting ACF and PACF: {e}")


def plot_sales_vs_customers(df):
    logger.info("Plotting sales vs customers scatter plot...")
    try:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df['Customers'], df['Sales'], c=df.index, cmap='viridis')
        plt.colorbar(scatter, label='Date')
        plt.title('Sales vs Customers Over Time')
        plt.xlabel('Number of Customers')
        plt.ylabel('Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting sales vs customers scatter plot: {e}")


def plot_store_type_performance(df):
    logger.info("Plotting store type performance...")
    try:
        store_type_sales = df.groupby([df.index.to_period('M'), 'StoreType'])['Sales'].mean().unstack()
        store_type_sales.plot(figsize=(15, 7))
        plt.title('Monthly Average Sales by Store Type')
        plt.xlabel('Date')
        plt.ylabel('Average Sales')
        plt.legend(title='Store Type')
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting store type performance: {e}")
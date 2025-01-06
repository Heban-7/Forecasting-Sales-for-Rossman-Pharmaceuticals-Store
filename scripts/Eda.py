import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    logging.info("Loading data from file...")
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    logging.info(f"Data loaded with shape {df.shape}")
    return df

def plot_sales_over_time(data, store_id):
    logging.info("Plotting Sales over Time for specific Store...")
    store_data = data[data['Store']== store_id]
    plt.figure(figsize=(12,7))
    plt.plot(store_data.index, store_data['Sales'], marker='o', linestyle='-', color='b')
    plt.title(f'Sales over Time for Store {store_id}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Time series plot for sales on all store
def plot_sales_over_time_all_store(data):
    logging.info('Plotting Sales over Time...')
    plt.figure(figsize=(12,7))
    plt.plot(data.index, data['Sales'], marker='o', linestyle='-', color='b')
    plt.title(f'Sales over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def weekly_sales_plot(df):
    logging.info("Weekly Sales Plot...")
    weekly_sales = df['Sales'].resample('W').sum()
    plt.figure(figsize=(15, 7))
    plt.plot(weekly_sales.index, weekly_sales)
    plt.title('Weekly Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.show()

def monthly_sales_sales(df):
    logging.info("Monthly Sales Plot...")
    monthly_sales = df['Sales'].resample('M').sum()
    plt.figure(figsize=(15, 7))
    plt.plot(monthly_sales.index, monthly_sales)
    plt.title('Monthly Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

def plot_yearly_sales(data):
    logging.info("Yearly Sales Plot...")
    df = data.copy()
    
    # Extract Year and Month data
    
    df['Year'] = df.index.year
    df['Month'] = df.index.month

    # Aggregate Sales by year and month
    df_plot = df.groupby(['Month', 'Year'])['Sales'].mean().reset_index()
    years = df_plot['Year'].unique()
    colors = plt.cm.tab10.colors

    # Plot
    plt.figure(figsize=(12,8))
    for i, y in enumerate(years):
        yearly_data = df_plot[df_plot['Year'] == y]
        plt.plot('Month', 'Sales', data=yearly_data, color=colors[i], label=y)
        plt.text(yearly_data['Month'].iloc[-1] + 0.1, yearly_data['Sales'].iloc[-1], str(y), fontsize=10, color=colors[i % len(colors)])
        
    plt.title('Seasonal Sales Plot by Month', fontsize=16)
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.legend(title='Year')
    plt.grid(True)
    plt.show()


def plot_weekly_sales(data):
    logging.info("Weekly Sales Plot Grouping by Month....")
    df = data.copy()
    # Extract Year, Month, and DayOfWeek
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['DayOfWeek'] = df['DayOfWeek']
        
    # Aggregate sales by Month and DayOfWeek
    df_plot = df.groupby(['Month', 'DayOfWeek'])['Sales'].mean().reset_index()
    day_mapping = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    df_plot['DayOfWeek'] = df_plot['DayOfWeek'].map(day_mapping)
        
    days = list(day_mapping.values())
    colors = plt.cm.tab10.colors
        
    # Plot
    plt.figure(figsize=(12, 8))
    for i, day in enumerate(days):  
        day_data = df_plot[df_plot['DayOfWeek'] == day]
        plt.plot('Month', 'Sales', data=day_data, color=colors[i % len(colors)], label=day)
        
    plt.title('Monthly Sales Grouped by Day of the Week', fontsize=16)
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.legend(title='Day of the Week')
    plt.grid(True)
    plt.show()


def seasonal_decomposition(df):
    logging.info("Performing seasonal decomposition...")
    monthly_sales = df['Sales'].resample('M').sum()
    result = seasonal_decompose(monthly_sales, model='additive')
    result.plot()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show() 

def plot_acf_pacf(df):
    logging.info("Plotting ACF and PACF...")
    monthly_sales = df['Sales'].resample('M').sum()
    n_lags = len(monthly_sales) // 3
    acf_values = acf(monthly_sales.dropna(), nlags=n_lags)
    pacf_values = pacf(monthly_sales.dropna(), nlags=n_lags)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    ax1.stem(range(len(acf_values)), acf_values)
    ax1.axhline(y=0, linestyle='--', color='gray')
    ax1.axhline(y=-1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax1.axhline(y=1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax1.set_title('Autocorrelation Function')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Correlation')

    ax2.stem(range(len(pacf_values)), pacf_values)
    ax2.axhline(y=0, linestyle='--', color='gray')
    ax2.axhline(y=-1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax2.axhline(y=1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax2.set_title('Partial Autocorrelation Function')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Correlation')

    plt.tight_layout()
    plt.show()


def plot_rolling_statistics(df):
    logging.info("Plotting rolling statistics...")
    monthly_sales = df['Sales'].resample('M').sum()
    rolling_mean = monthly_sales.rolling(window=12).mean()
    rolling_std = monthly_sales.rolling(window=12).std()

    plt.figure(figsize=(15, 7))
    plt.plot(monthly_sales.index, monthly_sales, label='Monthly Sales')
    plt.plot(rolling_mean.index, rolling_mean, label='12-month Rolling Mean')
    plt.plot(rolling_std.index, rolling_std, label='12-month Rolling Std')
    plt.legend()
    plt.title('Monthly Sales - Rolling Mean & Standard Deviation')
    plt.show()

def plot_day_of_week_sales(df):
    logging.info("Plotting average sales by day of week...")
    df = df.copy()
    df['DayOfWeek'] = df.index.dayofweek
    day_of_week_sales = df.groupby('DayOfWeek')['Sales'].mean()

    plt.figure(figsize=(10, 6))
    day_of_week_sales.plot(kind='bar')
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week (0=Monday.... 6=Sunday)')
    plt.ylabel('Average Sales')
    plt.show()

def plot_sales_through_month(df):
    logging.info("Plotting average sales by Month of Year...")
    df = df.copy()
    df['Month'] = df.index.month
    monthly_sales = df.groupby('Month')['Sales'].mean()

    plt.figure(figsize=(10, 6))
    monthly_sales.plot(kind='bar')
    plt.title('Average Sales Through the Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.show()

def box_plot_stateholiday_sales_distribution(df):
    logging.info("Plotting sales distribution During StateHoliday.... ")
    plt.figure(figsize=(10, 6))
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='StateHoliday', y='Sales', data=df)
    plt.title('Sales Distribution: Holiday vs Non-Holiday')
    plt.xlabel('Holiday (0 = None, a = public holiday, b = Easter holiday, c = Christmas)')
    plt.show()

def bar_plot_stateholiday_sales_distribution(df):
    logging.info("Bar Plotting sales distribution During StateHoliday.... ")
    stateholiday_sales = df.groupby('StateHoliday')['Sales'].mean()

    plt.figure(figsize=(10, 6))
    stateholiday_sales.plot(kind='bar')
    plt.title('State Holiday Sale Distribution')
    plt.xlabel('Holiday (0 = None, a = public holiday, b = Easter holiday, c = Christmas)')
    plt.ylabel('Average Sales')
    plt.xticks(rotation = 0)
    plt.show()

def plot_schoolholiday_sales_distribution(df):
    logging.info("Plotting sales distribution During SchoolHoliday.... ")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='SchoolHoliday', y='Sales', data=df)
    plt.title('Sales Distribution: Holiday vs Non-Holiday')
    plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
    plt.show()

def plot_promo_effect(df):
    logging.info("Plotting promo effect over time...")
    monthly_promo_sales = df.groupby([df.index.to_period('M'), 'Promo'])['Sales'].mean().unstack()
    monthly_promo_sales.columns = ['No Promo', 'Promo']

    monthly_promo_sales[['No Promo', 'Promo']].plot(figsize=(15, 7))
    plt.title('Monthly Average Sales: Promo vs No Promo')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(['No Promo', 'Promo'])
    plt.show()

def plot_store_type_performance(df):
    logging.info("Plotting store type performance over time...")
    store_type_sales = df.groupby([df.index.to_period('M'), 'StoreType'])['Sales'].mean().unstack()
    store_type_sales.plot(figsize=(15, 7))
    plt.title('Monthly Average Sales by Store Type')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(title='Store Type')
    plt.show()

def plot_assortment_type_performance(df):
    logging.info("Plotting Assortment type performance over time...")
    store_type_sales = df.groupby([df.index.to_period('M'), 'Assortment'])['Sales'].mean().unstack()
    store_type_sales.plot(figsize=(15, 7))
    plt.title('Monthly Average Sales by Assortment')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(title='Assortment')
    plt.show()

def plot_sales_vs_customers(df):
    logging.info("Plotting sales vs customers scatter plot...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Customers'], df['Sales'], c=df.index, cmap='viridis')
    plt.colorbar(scatter, label='Date')
    plt.title('Sales vs Customers Over Time')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()


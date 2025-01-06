import pandas as pd

def summary_statistics(df):
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
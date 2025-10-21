import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_credit(path='data/sample_credit.csv'):
    df = pd.read_csv(path)
    return df

def load_prices(path='data/sample_prices.csv'):
    df = pd.read_csv(path, parse_dates=['date']).set_index('date')
    return df

def train_test_split_credit(df, target='defaulted', test_size=0.2, random_state=42):
    X = df.drop(columns=['id', target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

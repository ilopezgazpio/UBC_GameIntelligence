import os
import pandas as pd


def load_dataset(name: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', name + '.csv')
    df = pd.read_csv(path).sort_values('Date')
    return df
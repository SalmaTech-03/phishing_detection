import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/phishing_dataset.csv")

df['Result'] = df['Result'].replace({-1: 0, 1: 1})

X = df.drop('Result', axis=1)
y = df['Result']

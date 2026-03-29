import pandas as pd

data = pd.read_csv("data/landmark_data.csv", header=None)
print(data.iloc[:, -1].value_counts())
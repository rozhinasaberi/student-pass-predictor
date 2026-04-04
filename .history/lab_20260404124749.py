import pandas as pd

data_rojina = pd.read_csv("student-por.csv", sep=";")

print(data_rojina.head())
print(data_rojina.shape)
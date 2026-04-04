import pandas as pd

data_rojina = pd.read_csv("student-por.csv", sep=";")

print(data_rojina.head())
print(data_rojina.shape)

print("first 5 rows:")
print(data_rojina.head())
print()

print("shape:")
print(data_rojina.shape)
print()

print("column names:")
print(data_rojina.columns)
print()

print("data types:")
print(data_rojina.dtypes)
print()

print("missing values:")
print(data_rojina.isnull().sum())
print()

print("numeric statistics:")
print(data_rojina.describe())
print()

print("categorical values:")
for col in data_rojina.select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(data_rojina[col].value_counts())
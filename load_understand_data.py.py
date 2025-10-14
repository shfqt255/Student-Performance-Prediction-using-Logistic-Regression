import pandas as pd

data_frame=pd.read_csv("student_performance.csv")

print(f"Sample Rows: \n {data_frame.head()} \n")

print(f"Data set Shape: \n     Rows: {data_frame.shape[0]},  Columns: {data_frame.shape[1]} \n ")

print(f"Data set Info: ")
data_frame.info()
print("")

print(f"Summary Statistics: \n {data_frame.describe(include='all')} \n")

print(f"Total Missing Values: \n {data_frame.isnull().sum()}\n")

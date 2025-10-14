import pandas as pd
from sklearn.preprocessing import LabelEncoder

data_frame=pd.read_csv("student_performance.csv")

label_encoder=LabelEncoder()

data_frame["Internet"]=label_encoder.fit_transform(data_frame["Internet"])
data_frame["Passed"]=label_encoder.fit_transform(data_frame["Passed"])

print("Missing Values: ")
print(data_frame.isnull().sum())

print("Data Types of the data: ")
print(data_frame.dtypes)

print(data_frame.head())

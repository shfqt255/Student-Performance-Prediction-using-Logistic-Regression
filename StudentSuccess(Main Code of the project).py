import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


data_frame=pd.read_csv("student_performance.csv")

label_encoder=LabelEncoder()
data_frame["Internet"]=label_encoder.fit_transform(data_frame["Internet"])
data_frame["Passed"]=label_encoder.fit_transform(data_frame["Passed"])

standard_scaler=StandardScaler()
features=["StudyHours", "Attendance","PastScore","Internet","SleepHours","Passed"]
data_frame_copy=data_frame.copy()
data_frame_copy[features]=data_frame_copy[features]

X=data_frame_copy[features]
Y=data_frame_copy["Passed"]
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.2, random_state=42)
modal=LogisticRegression()
modal.fit(X_train, Y_train)
Y_prediction=modal.predict(X_test)

print("Classification Report: ")
print(classification_report(Y_test, Y_prediction))

conf_matrix=confusion_matrix(Y_test, Y_prediction)
# This will show a matrix for only four dataset values. Because I have ginven 20% of the data for test. 
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

try:
    study_hours=float(input("Enter your Study hours: "))
    attendance=float(input("Enter your attendance: "))
    sleep_hours=float(input("Enter your Sleep hours: "))
    past_score=float(input("Enter your Past Score: "))

    user_data_frame=pd.DataFrame(
        [{
            "StudyHours": study_hours,
            "Attendance" : attendance,
            "SleepHours": sleep_hours,
            "PastScore": past_score
        }]
    )

    user_input_scaled=standard_scaler.fit_transform(user_data_frame)
    user_input_prediction=modal.predict(user_input_scaled)
    result="Pass" if user_input_prediction==1 else "Fail"

    print(f"Prediction based on the user input: {result}")
except Exception as e:
    print(f"Error:{e}")
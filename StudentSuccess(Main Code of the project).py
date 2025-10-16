import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_frame = pd.read_csv("student_performance.csv")

# Encode categorical columns
label_encoder = LabelEncoder()
data_frame["Internet"] = label_encoder.fit_transform(data_frame["Internet"])
data_frame["Passed"] = label_encoder.fit_transform(data_frame["Passed"])

# Features and labels
features = ["StudyHours", "Attendance", "PastScore", "Internet", "SleepHours"]
X = data_frame[features]
Y = data_frame["Passed"]

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# Predict and evaluate
Y_prediction = model.predict(X_test_scaled)

print("Classification Report:\n")
print(classification_report(Y_test, Y_prediction))

# Confusion matrix visualization
conf_matrix = confusion_matrix(Y_test, Y_prediction)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# User input prediction
try:
    study_hours = float(input("Enter your Study hours: "))
    attendance = float(input("Enter your attendance: "))
    sleep_hours = float(input("Enter your Sleep hours: "))
    past_score = float(input("Enter your Past Score: "))
    internet = int(input("Do you have Internet? (1 for Yes, 0 for No): "))

    user_data = pd.DataFrame([{
        "StudyHours": study_hours,
        "Attendance": attendance,
        "PastScore": past_score,
        "Internet": internet,
        "SleepHours": sleep_hours
    }])

    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)

    result = "Pass" if prediction[0] == 1 else "Fail"
    print(f"\nPrediction based on your input: {result}")
except Exception as e:
    print(f"Error: {e}")

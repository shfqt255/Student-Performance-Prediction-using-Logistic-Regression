# 🎓 Student Performance Prediction using Logistic Regression

A **Machine Learning classification model** built using **Scikit-learn** that predicts a student’s performance (Pass or Fail) based on study habits, attendance, sleep duration, and past scores.  
The project demonstrates the complete ML workflow — from data preprocessing and model training to evaluation and visualization using a confusion matrix.

---

## 🧩 Key Features
✅ Data preprocessing using **LabelEncoder** and **StandardScaler**  
✅ Train-test split for unbiased model evaluation  
✅ Model training with **Logistic Regression**  
✅ Model evaluation using **classification report** and **confusion matrix**  
✅ Visualization using **Seaborn heatmap**  
✅ User input section for live predictions  

---

## 📊 Dataset Columns

| Column Name | Description |
|--------------|-------------|
| StudyHours | Number of hours studied daily |
| Attendance | Attendance percentage |
| SleepHours | Average sleep duration |
| PastScore | Previous academic score |
| Internet | Whether the student has internet access (Yes/No) |
| Passed | Target variable — whether the student passed (Yes/No) |

---

## ⚙️ Libraries Used
```python
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 🚀 How It Works

1. **Data Preprocessing:**  
   - Encodes categorical variables (`Internet`, `Passed`)  
   - Normalizes numerical features for uniform scaling  

2. **Model Training:**  
   - Splits data (80% training, 20% testing)  
   - Trains a Logistic Regression model on training data  

3. **Model Evaluation:**  
   - Uses classification report and confusion matrix for accuracy, precision, and recall  

4. **Prediction:**  
   - Accepts user input for new data  
   - Predicts if the student will Pass or Fail  

---

## 📈 Output Example

```
Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.80      0.82        10
           1       0.88      0.92      0.90        15

    accuracy                           0.87        25
   macro avg       0.86      0.86      0.86        25
weighted avg       0.87      0.87      0.87        25
```
✅ Confusion matrix visualized using Seaborn.

---

## 🧮 User Prediction Example
```
Enter your Study hours: 5
Enter your attendance: 88
Enter your Sleep hours: 7
Enter your Past Score: 75
Prediction based on the user input: Pass
```

---



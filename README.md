# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare Dataset
2. Split the Dataset
3. Train the Decision Tree Model
4. Evaluate and Visualize Results 


## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("tumor.csv")
print(data.head())
print(data.columns)
x=data.drop(columns=['Class'])
y=data['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model= DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Name: A.Jannathul Shaban")
print("Register Number:212225220043")
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
classification=classification_report(y_test,y_pred)
print("Classification Report:",classification)
confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:",confusion)
sns.heatmap(confusion,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="821" height="321" alt="image" src="https://github.com/user-attachments/assets/37ab979d-446e-4849-9ba8-174e1801b418" />

<img width="680" height="576" alt="image" src="https://github.com/user-attachments/assets/831b993c-66fa-41ff-91ae-51613c18e36e" />


## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.

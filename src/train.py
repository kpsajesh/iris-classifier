#pip install --upgrade pip
#pip install numpy pandas scikit-learn matplotlib seaborn 
#python -c "import matplotlib as m; print(m.__version__)"

# run from CLI
# 1. python -m venv venv
# 2. .\venv\Scripts\Activate.ps1  
# 3. pip install numpy pandas scikit-learn matplotlib seaborn
# 4. python src/train.py

import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
print(iris.feature_names, iris.target_names)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#print("Predictions:", y_pred[:5])
#print("True labels:", y_test[:5])

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("classification_report:", report)
print('Confusion Matrix')
print(conf_mat)


output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "model_results.txt")

with open(output_file, "w") as f:
    f.write("Iris Classification Results\n")
    f.write("---------------------------\n")    
    f.write(f"Accuracy: {accuracy}\n\n")
    
    f.write("\nClassification Report:\n")
    f.write("----------------------\n")
    f.write(report + "\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(conf_mat))

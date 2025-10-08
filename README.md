"""
🌸 Iris Flower Classification using K-Nearest Neighbors (KNN)
Author: Your Name
GitHub: https://github.com/your-username
---------------------------------------------
This all-in-one script:
1️⃣ Trains a KNN classifier on the Iris dataset.
2️⃣ Evaluates model accuracy.
3️⃣ Predicts the class of a custom sample.
4️⃣ Prints a built-in README summary.
"""

# ---------- Import Required Libraries ----------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import textwrap
import os

# ---------- Step 1: Load Dataset ----------
iris = load_iris()
X, y = iris.data, iris.target  # features and labels

# ---------- Step 2: Split Dataset ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Step 3: Initialize & Train Model ----------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ---------- Step 4: Evaluate Model ----------
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ---------- Step 5: Predict a New Sample ----------
sample = [[5.1, 3.5, 1.4, 0.2]]  # example flower measurements
predicted_class = iris.target_names[knn.predict(sample)[0]]

print("\n🔍 Sample Input:", sample)
print("🌼 Predicted Species:", predicted_class)

# ---------- Step 6: Create requirements.txt ----------
requirements = """scikit-learn==1.5.0
numpy==1.26.4
"""
with open("requirements.txt", "w") as f:
    f.write(requirements)
print("\n📦 requirements.txt created successfully!")

# ---------- Step 7: Create README.md ----------
readme_content = textwrap.dedent(f"""
# 🌸 Iris Flower Classification using K-Nearest Neighbors (KNN)

This project demonstrates a simple **Machine Learning classification** task using the **Iris dataset** from scikit-learn.
It uses the **K-Nearest Neighbors (KNN)** algorithm to classify iris flowers into three species based on four input features.

---

## 📊 Dataset

The dataset contains **150 samples** from **three species** of iris flowers:
- Setosa
- Versicolor
- Virginica

Each sample has four features:
1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

---

## ⚙️ Requirements

Install dependencies:
```bash
pip install -r requirements.txt

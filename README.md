# 🏦 Bank Customer Churn Prediction using Artificial Neural Network (ANN)

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-ANN-red?style=for-the-badge&logo=keras)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

This project builds an **end-to-end Artificial Neural Network (ANN)** to predict whether a bank customer will **churn (leave the bank)** based on their profile and account information.

The dataset contains **10,000 real customer records** with 14 features including CreditScore, Geography, Age, Balance, and more.

A key challenge in this project was **class imbalance** — only 20% of customers churned. This was solved using **SMOTE (Synthetic Minority Oversampling Technique)**, which significantly improved the model's fairness and real-world usefulness.

---

## 🎯 Problem Statement

> Banks lose millions of dollars every year due to customer churn. Predicting which customers are likely to leave allows banks to take proactive action and retain them.

**Target Variable:** `Exited`
- `1` → Customer churned (left the bank)
- `0` → Customer stayed

---

## 📂 Project Structure

```
bank-churn-prediction/
│
├── Churn_Modelling.csv          # Dataset
├── bank_churn_prediction.ipynb  # Main Jupyter Notebook
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies
```

---

## 📊 Dataset Information

| Feature | Description | Type |
|---|---|---|
| RowNumber | Row index | Dropped |
| CustomerId | Unique customer ID | Dropped |
| Surname | Customer surname | Dropped |
| CreditScore | Credit score of customer | Numerical |
| Geography | Country (France/Spain/Germany) | Categorical |
| Gender | Male or Female | Categorical |
| Age | Age of customer | Numerical |
| Tenure | Years with bank | Numerical |
| Balance | Account balance | Numerical |
| NumOfProducts | Number of products used | Numerical |
| HasCrCard | Has credit card? (0/1) | Binary |
| IsActiveMember | Active member? (0/1) | Binary |
| EstimatedSalary | Estimated annual salary | Numerical |
| **Exited** | **Churned? (0/1) — TARGET** | **Binary** |

- **Total Records:** 10,000
- **Total Features:** 14 (11 after cleaning)
- **Missing Values:** None

---

## 🔄 Project Workflow

```
Raw Data
    ↓
Exploratory Data Analysis (EDA)
    ↓
Data Cleaning & Preprocessing
    ↓
Feature Encoding (Label + One-Hot)
    ↓
Feature Scaling (MinMaxScaler)
    ↓
Handle Class Imbalance (SMOTE)
    ↓
Train/Test Split
    ↓
Build ANN Model
    ↓
Train 100 Epochs
    ↓
Evaluate (Confusion Matrix + Report)
```

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked dataset shape, data types, null values using `df.info()` and `df.describe()`
- Analyzed unique value counts per column
- Visualized **Tenure distribution** for churned vs non-churned customers
- Visualized **EstimatedSalary distribution** for churned vs non-churned customers
- Key finding: **EstimatedSalary has no impact on churn** — uniform distribution across both classes
- Key finding: **Tenure is slightly more predictive** of churn behavior

---

## 🧹 Data Preprocessing

### 1. Drop Irrelevant Columns
```python
df.drop(['RowNumber', 'CustomerId', 'Surname'], inplace=True, axis=1)
```

### 2. Label Encoding — Gender
```python
df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
```

### 3. One-Hot Encoding — Geography
```python
df = pd.get_dummies(df, columns=['Geography'], dtype=int)
# Creates: Geography_France, Geography_Germany, Geography_Spain
```

### 4. Feature Scaling — MinMaxScaler
```python
from sklearn.preprocessing import MinMaxScaler

cols_to_scale = ['CreditScore', 'Age', 'Tenure', 'Balance',
                 'NumOfProducts', 'EstimatedSalary']
scaler = MinMaxScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])
```

---

## ⚖️ Handling Class Imbalance with SMOTE

The dataset had a significant class imbalance:

| Class | Count | Percentage |
|---|---|---|
| 0 — Stayed | 7,963 | 79.6% |
| 1 — Churned | 2,037 | 20.4% |

This caused the initial model to be **biased toward predicting "stayed"**, achieving high accuracy but poor churn detection.

**Solution: SMOTE (Synthetic Minority Oversampling Technique)**

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)

# After SMOTE:
# Class 0: 7,963
# Class 1: 7,963  ← balanced!
```

---

## 🧠 ANN Architecture

```
Input Layer  →  12 features
      ↓
Hidden Layer 1  →  20 neurons  |  ReLU activation
      ↓
Hidden Layer 2  →  15 neurons  |  ReLU activation
      ↓
Output Layer  →  1 neuron  |  Sigmoid activation
```

```python
model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(12,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1,  activation='sigmoid'),
])

model.compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy']
)

model.fit(X_train, y_train, epochs=100)
```

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Output Activation | Sigmoid |
| Hidden Activation | ReLU |
| Epochs | 100 |
| Input Features | 12 |

---

## 📈 Results

### Before SMOTE

| Metric | Class 0 (Stayed) | Class 1 (Churned) |
|---|---|---|
| Precision | 0.91 | 0.56 |
| Recall | 0.87 | 0.67 |
| F1-Score | 0.89 | 0.61 |
| **Accuracy** | **83%** | |

**Problem:** Model performed well on majority class but poorly on minority (churned) class.

---

### After SMOTE ✅

| Metric | Class 0 (Stayed) | Class 1 (Churned) |
|---|---|---|
| Precision | 0.83 | 0.79 |
| Recall | 0.78 | 0.83 |
| F1-Score | 0.81 | 0.81 |
| **Accuracy** | **81%** | |

**Confusion Matrix (After SMOTE):**

```
                  Predicted 0    Predicted 1
Actual 0  →          1280            352
Actual 1  →           260           1294
```

✅ **81% balanced accuracy with equal performance on both classes**

---

## 💡 Key Learnings

- **Accuracy can be misleading** — always check Precision, Recall, and F1-Score
- **Class imbalance is a real problem** — SMOTE is an effective solution
- **Confusion matrix tells the real story** — not just a single accuracy number
- **Data preprocessing matters more than model complexity**
- **fit_transform on train only** — only transform() on test data to avoid data leakage

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Programming language |
| TensorFlow / Keras | ANN model building |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing + Metrics |
| imbalanced-learn | SMOTE implementation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/bank-churn-prediction.git
cd bank-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook bank_churn_prediction.ipynb
```

---

## 📦 Requirements

```
tensorflow>=2.10
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
jupyter
```



## 👤 Author

**Your Name**
- LinkedIn: [https://www.linkedin.com/in/nazikhassan11]

- Email: nasikmohamednzk2020@gmail.com

---

## ⭐ If you found this project helpful, please give it a star!

```
git clone → pip install → jupyter notebook → learn!
```

---

*Built with passion for learning Data Science and Machine Learning* 🚀

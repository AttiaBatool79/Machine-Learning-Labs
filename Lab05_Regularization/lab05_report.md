# Lab 5: Overfitting and Regularization in Linear Regression

## Table of Contents
1. [Objective](#objective)
2. [Introduction](#introduction)
3. [Dataset Overview](#dataset-overview)
4. [Python Libraries](#python-libraries)
5. [Steps for Regularized Linear Regression](#steps-for-regularized-linear-regression)
   - [1. Load the Dataset](#1-load-the-dataset)
   - [2. Data Preprocessing](#2-data-preprocessing)
   - [3. Splitting the Data](#3-splitting-the-data)
   - [4. Training Standard Linear Regression](#4-training-standard-linear-regression)
   - [5. Applying Ridge Regression](#5-applying-ridge-regression)
   - [6. Applying Lasso Regression](#6-applying-lasso-regression)
   - [7. Evaluating the Models](#7-evaluating-the-models)
   - [8. Visualizing Results](#8-visualizing-results)
6. [Lab Questions and Answers](#lab-questions-and-answers)

---

## Objective
The goal of this lab is to:

1. Understand overfitting in machine learning models.
2. Implement L1 (Lasso) and L2 (Ridge) regularization.
3. Compare model performances to analyze the effect of regularization.

---

## Introduction
Regularization techniques help improve model generalization by reducing overfitting. The two commonly used techniques are:
- **Ridge Regression (L2)**: Shrinks coefficients but does not eliminate them.
- **Lasso Regression (L1)**: Can shrink some coefficients to zero, performing feature selection.

---

## Dataset Overview
The dataset contains housing market information with features such as:

| Column | Description |
|--------|-------------|
| **Suburb** | Location of the property |
| **Rooms** | Number of rooms |
| **Price** | Target variable (house price) |

---

## Python Libraries
We will use the following libraries:
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib & seaborn**: Visualization
- **scikit-learn**: Model training & evaluation

Install them using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Steps for Regularized Linear Regression

### 1. Load the Dataset
```python
import pandas as pd

# Load dataset
dataset = pd.read_csv("Melbourne_housing_FULL.csv")
print(dataset.head())
```

### 2. Data Preprocessing
```python
# Selecting relevant features
cols_to_use = ['Rooms', 'Price']
dataset = dataset[cols_to_use]

# Handling missing values
dataset.fillna(dataset.mean(), inplace=True)
```

### 3. Splitting the Data
```python
from sklearn.model_selection import train_test_split

X = dataset[['Rooms']]
y = dataset['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
```

### 4. Training Standard Linear Regression
```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
```

### 5. Applying Ridge Regression
```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=50)
ridge_reg.fit(X_train, y_train)
```

### 6. Applying Lasso Regression
```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=50)
lasso_reg.fit(X_train, y_train)
```

### 7. Evaluating the Models
```python
from sklearn.metrics import r2_score

print("Linear Regression R² Score:", r2_score(y_test, reg.predict(X_test)))
print("Ridge Regression R² Score:", r2_score(y_test, ridge_reg.predict(X_test)))
print("Lasso Regression R² Score:", r2_score(y_test, lasso_reg.predict(X_test)))
```

### 8. Visualizing Results
```python
import matplotlib.pyplot as plt
import seaborn as sns

models = ['Linear', 'Ridge', 'Lasso']
scores = [r2_score(y_test, reg.predict(X_test)), 
          r2_score(y_test, ridge_reg.predict(X_test)), 
          r2_score(y_test, lasso_reg.predict(X_test))]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=scores)
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("Comparison of Regression Models")
plt.show()
```

---
## Deliverables
1. A Jupyter Notebook with full implementation.
2. Visualizations of regression results.
3. Answers to lab questions.

## Getting Started as a Beginner
If you're new to regression and Python, start by:
1. **Installing Python and Jupyter Notebook**: Use Anaconda or install Python and `pip install jupyter`.
2. **Installing Libraries**: Run `pip install pandas numpy matplotlib seaborn scikit-learn`.
3. **Running the Provided Code**: Copy and paste each section into a Jupyter Notebook and execute it step by step.
4. **Understanding Outputs**: Read through the printed outputs and visualizations to grasp what each step is doing.
5. **Experimenting**: Change some values, try adding new variables, and explore different datasets to get comfortable.



# Lab 6: Polynomial Regression

## Table of Contents
1. [Objective](#objective)
2. [Introduction](#introduction)
3. [Dataset Overview](#dataset-overview)
4. [Python Libraries](#python-libraries)
5. [Steps for Polynomial Regression](#steps-for-polynomial-regression)
   - [1. Load the Dataset](#1-load-the-dataset)
   - [2. Data Preprocessing](#2-data-preprocessing)
   - [3. Splitting the Data](#3-splitting-the-data)
   - [4. Training Linear Regression](#4-training-linear-regression)
   - [5. Training Polynomial Regression](#5-training-polynomial-regression)
   - [6. Predicting Results](#6-predicting-results)
   - [7. Visualizing Results](#7-visualizing-results)
6. [Deliverables](#deliverables)
7. [Getting Started as a Beginner](#getting-started-as-a-beginner)

---

## Objective
The goal of this lab is to:
1. Understand Polynomial Regression and its advantages over Linear Regression.
2. Implement Polynomial Regression using Python.
3. Compare model performances to analyze the effect of polynomial degrees.

---

## Introduction
Polynomial Regression is an extension of Linear Regression where the relationship between the independent variable (X) and dependent variable (y) is modeled as an **nth-degree polynomial**. It helps in capturing **non-linear** patterns in data.

The general equation for Polynomial Regression is:

\[ y = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3 + ... + \beta_nx^n \]

Where:
- **y** = Dependent variable
- **x** = Independent variable
- **\( \beta_0, \beta_1, ..., \beta_n \)** = Coefficients
- **n** = Degree of the polynomial

---

## Dataset Overview
The dataset contains information about job levels and salaries.

| Column | Description |
|--------|-------------|
| **Position Level** | Independent variable representing experience level |
| **Salary** | Target variable (salary amount) |

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

## Steps for Polynomial Regression

### 1. Load the Dataset
```python
import pandas as pd

# Load dataset
df = pd.read_csv('Position_Salaries.csv')
print(df.head())
```

### 2. Data Preprocessing
```python
# Extracting relevant features
X = df.iloc[:, 1:-1].values  # Independent variable
y = df.iloc[:, -1].values    # Dependent variable
```

### 3. Splitting the Data
(Not required as we have a small dataset)

### 4. Training Linear Regression
```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X, y)
```

### 5. Training Polynomial Regression
```python
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=4)
X_poly = pr.fit_transform(X)

lr_poly = LinearRegression()
lr_poly.fit(X_poly, y)
```

### 6. Predicting Results
```python
# Linear Regression Prediction
y_pred_lr = lr.predict(X)

# Polynomial Regression Prediction
y_pred_poly = lr_poly.predict(X_poly)
```

### 7. Visualizing Results
```python
import matplotlib.pyplot as plt
import numpy as np

# Linear Regression Plot
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred_lr, color='red')
plt.title('Linear Regression Fit')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Polynomial Regression Plot
X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
plt.scatter(X, y, color='blue')
plt.plot(X_grid, lr_poly.predict(pr.fit_transform(X_grid)), color='red')
plt.title('Polynomial Regression Fit')
plt.xlabel('Position Level')
plt.ylabel('Salary')
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

---

✅ **Polynomial Regression helps in capturing complex relationships that Linear Regression might miss!** 🚀



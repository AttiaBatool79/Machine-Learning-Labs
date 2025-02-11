# Lab 3: Linear Regression with One Variable

## Table of Contents
1. [Objective](#objective)
2. [Introduction](#introduction)
3. [Dataset Overview](#dataset-overview)
4. [Python Libraries](#python-libraries)
5. [Steps for Linear Regression](#steps-for-linear-regression)
   - [1. Load the Dataset](#1-load-the-dataset)
   - [2. Data Preprocessing](#2-data-preprocessing)
   - [3. Splitting the Data](#3-splitting-the-data)
   - [4. Training the Model](#4-training-the-model)
   - [5. Making Predictions](#5-making-predictions)
   - [6. Evaluating the Model](#6-evaluating-the-model)
   - [7. Visualizing Results](#7-visualizing-results)
6. [Lab Questions and Answers](#lab-questions-and-answers)

---

## Objective
The goal of this lab is to understand and implement **Simple Linear Regression** using one independent variable to predict an outcome.

1. Learn how to fit a linear model.
2. Interpret model coefficients.
3. Evaluate performance using key metrics.
4. Visualize results using regression plots.

---

## Introduction
Linear regression is a fundamental machine learning algorithm used to predict a dependent variable based on an independent variable. The relationship is represented by:

\[ y = \beta_0 + \beta_1x + \epsilon \]

Where:
- **y**: Dependent variable (target)
- **x**: Independent variable (predictor)
- **β₀**: Intercept
- **β₁**: Slope
- **ε**: Error term

---

## Dataset Overview
The dataset contains housing information with features such as:

| Column | Description |
|--------|-------------|
| **Avg. Area House Age** | Age of houses in the region |
| **Price** | House price (dependent variable) |

---

## Python Libraries
We will use the following libraries:
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib**: Visualization
- **scikit-learn**: Model training & evaluation

Install them using:
```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## Steps for Linear Regression

### 1. Load the Dataset
```python
import pandas as pd

# Load dataset
data = pd.read_csv("housing.csv")
print(data.head())
```

### 2. Data Preprocessing
```python
# Selecting independent and dependent variables
X = data[['Avg. Area House Age']]
y = data['Price']
```

### 3. Splitting the Data
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Training the Model
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### 5. Making Predictions
```python
y_pred = model.predict(X_test)
```

### 6. Evaluating the Model
```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
```

### 7. Visualizing Results
```python
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel("Avg. Area House Age")
plt.ylabel("Price")
plt.title("Linear Regression: House Age vs Price")
plt.legend()
plt.show()
```

---

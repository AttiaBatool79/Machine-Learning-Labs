# Lab 4: Linear Regression with Multiple Variables

## Table of Contents

[Objective](#objective)

[Prerequisites](#prerequisites)

[Introduction to Multiple Linear Regression](#introduction-to-multiple-linear-regression)

[Assumptions of Multiple Linear Regression](#assumptions-of-multiple-linear-regression)

[Evaluating Multiple Linear Regression Models](#evaluating-multiple-linear-regression-models)

[Visualization](#visualization)

[Implementation of Multiple Linear Regression](#implementation-of-multiple-linear-regression)

  - [Step 1: Import Libraries](#step-1-import-libraries)
  - [Step 2: Load Dataset](#step-2-load-dataset)
  - [Step 3: Data Preprocessing](#step-3-data-preprocessing)
  - [Step 4: Train-Test Split](#step-4-train-test-split)
  - [Step 5: Train the Model](#step-5-train-the-model)
  - [Step 6: Make Predictions](#step-6-make-predictions)
  - [Step 7: Evaluate the Model](#step-7-evaluate-the-model)
  - [Step 8: Visualize the Results](#step-8-visualize-the-results)

[Summary and Insights](#summary-and-insights)

[Deliverables](#deliverables)

[Lab Questions](#lab-questions)

[Getting Started as a Beginner](#getting-started-as-a-beginner)

## Objective
The goal of this lab is to understand and implement Multiple Linear Regression, where multiple independent variables influence the dependent variable. This includes:

- Understanding the core concepts and assumptions.
- Implementing Multiple Linear Regression using Python.
- Visualizing relationships and interpreting results.
- Evaluating model performance using key metrics.

## Prerequisites
- Basic Python programming knowledge.
- Familiarity with Pandas, NumPy, Matplotlib, and Scikit-learn.
- Understanding of Simple Linear Regression and its assumptions.

## Introduction to Multiple Linear Regression
**What is Regression?**
Regression is a statistical method for modeling relationships between a dependent variable (target) and multiple independent variables (predictors).

## Assumptions of Multiple Linear Regression
1. **Linearity**: The relationship between x and y is linear.
2. **Independence**: Observations are independent.
3. **Homoscedasticity**: Residuals have constant variance.
4. **Normality of Residuals**: Residuals follow a normal distribution.

## Evaluating Multiple Linear Regression Models
**Key Metrics:**
- **Mean Squared Error (MSE)**: Measures the average squared error between predicted and actual values.
- **R-squared (R²)**: Explains the proportion of variance in y explained by x variables.

## Visualization
- **Regression Line (Multi-dimensions)**: Helps assess model performance.
- **Residual Plot**: Checks for assumption violations.

## Implementation of Multiple Linear Regression
### Step 1: Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

### Step 2: Load Dataset
```python
data = pd.read_csv("housing.csv")  
print("First 5 rows of the dataset:")  
print(data.head())  
```

### Step 3: Data Preprocessing
```python
X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',  
          'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']  
print("\nMissing values in the dataset:")  
print(data.isnull().sum())  
```

### Step 4: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
```

### Step 5: Train the Model
```python
model = LinearRegression()  
model.fit(X_train, y_train)  
print("Intercept (β0):", model.intercept_)  
print("Coefficients (β1, β2, ..., βn):", model.coef_)  
```

### Step 6: Make Predictions
```python
y_pred = model.predict(X_test)  
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})  
print(comparison.head())  
```

### Step 7: Evaluate the Model
```python
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  
print("\nModel Evaluation Metrics:")  
print("Mean Squared Error (MSE):", mse)  
print("R-squared (R²):", r2)  
```

### Step 8: Visualize the Results
```python
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()
```

## Summary and Insights
- **Intercept (β0)**: Predicted price when all independent variables are zero.
- **Coefficients (β1, β2, ..., βn)**: Measure the impact of each variable on price.
- **R-squared (R²)**: Evaluates how well the model explains variations in price.
- **Visualization**: Helps verify model performance.

## Deliverables
1. A Jupyter Notebook with full implementation.
2. Visualizations of regression results.
3. Answers to lab questions.

## Lab Questions
1. What does each coefficient (β1, β2, ...) indicate for this dataset?
2. How well does the model predict Price based on R²?
3. Are there any patterns in the residuals that violate regression assumptions?

## Getting Started as a Beginner
If you're new to regression and Python, start by:
1. **Installing Python and Jupyter Notebook**: Use Anaconda or install Python and `pip install jupyter`.
2. **Installing Libraries**: Run `pip install pandas numpy matplotlib seaborn scikit-learn`.
3. **Running the Provided Code**: Copy and paste each section into a Jupyter Notebook and execute it step by step.
4. **Understanding Outputs**: Read through the printed outputs and visualizations to grasp what each step is doing.
5. **Experimenting**: Change some values, try adding new variables, and explore different datasets to get comfortable.

Happy coding! 🚀

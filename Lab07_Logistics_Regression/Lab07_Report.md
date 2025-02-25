# Lab 7 - Logistic Regression 

## Table of Contents
- [What I Did](#what-i-did)
- [What is Logistic Regression?](#what-is-logistic-regression)
  - [How Does It Work?](#how-does-it-work)
- [What I Used](#what-i-used)
- [Steps I Followed](#steps-i-followed)
  1. [Import Tools](#1-import-tools)
  2. [Load and See the Data](#2-load-and-see-the-data)
  3. [Split Data into Two Parts](#3-split-data-into-two-parts)
  4. [Train the Model](#4-train-the-model)
  5. [Make Predictions](#5-make-predictions)
  6. [Check If My Model is Good](#6-check-if-my-model-is-good)
  7. [See How the Model Thinks](#7-see-how-the-model-thinks)
  8. [Make My Own Predictions](#8-make-my-own-predictions)
- [What I Learned](#what-i-learned)
- [Extra Things I Can Try](#extra-things-i-can-try)

## What I Did
I learned how to use **Logistic Regression** to predict if someone will buy insurance based on their age. This is a way for computers to make smart guesses!

## What is Logistic Regression?
It is a type of math that helps us decide between two choices, like **yes or no**. Here, we use it to guess if a person will buy insurance (1 = Yes, 0 = No).

### How Does It Work?
- It takes a number (like age) and does some math.
- It uses a special formula called the **sigmoid function** to give a number between 0 and 1.
- If the number is **greater than 0.5**, we say **YES**.
- If it's **less than 0.5**, we say **NO**.

## What I Used
I used these tools to do my work:
- **pandas** - To handle data
- **numpy** - To do math
- **matplotlib** - To make charts
- **scikit-learn** - To train and test my model

## Steps I Followed

### 1. Import Tools
I first added all the tools I needed.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### 2. Load and See the Data
I opened a file called `insurance_data.csv` and looked at the first few rows.
```python
df = pd.read_csv("insurance_data.csv")
print(df.head())
```
Then I made a **scatter plot** to see how age and insurance buying are related.
```python
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
plt.xlabel("Age")
plt.ylabel("Bought Insurance")
plt.show()
```

### 3. Split Data into Two Parts
I split my data into **training data** (80%) and **testing data** (20%).
```python
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, train_size=0.8, random_state=42)
```

### 4. Train the Model
I made the computer learn from the training data.
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 5. Make Predictions
I let the computer guess results for the test data.
```python
y_pred = model.predict(X_test)
print("Predictions:", y_pred)
```

### 6. Check If My Model is Good
I checked how many answers were correct.
```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
```

### 7. See How the Model Thinks
I looked at the numbers the model uses to make decisions.
```python
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
```

### 8. Make My Own Predictions
I created a function to guess if someone will buy insurance.
```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def prediction_function(age):
    z = model.coef_[0] * age + model.intercept_
    y = sigmoid(z)
    return y

print("Prediction for age 35:", prediction_function(35))
print("Prediction for age 43:", prediction_function(43))
```

## What I Learned
- Logistic regression helps computers **guess between two choices**.
- It uses a **formula** to give a number between **0 and 1**.
- If the number is **big**, we say **yes**; if small, we say **no**.

## Extra Things I Can Try
- Change the train-test split from **80-20** to **70-30** and see if the model gets better.
- Use `model.predict_proba(X_test)` to see probabilities instead of just yes/no.
- Try predicting for new ages and see what happens!

---




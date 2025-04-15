# Lab 8: Open Ended Lab - Employee Attrition Classification

## Table of Contents
1. [Objective](#objective)
2. [Introduction](#introduction)
3. [Dataset Overview](#dataset-overview)
4. [Python Libraries](#python-libraries)
5. [Steps for Classification](#steps-for-classification)
   - [1. Load the Dataset](#1-load-the-dataset)
   - [2. Data Preprocessing](#2-data-preprocessing)
   - [3. Feature Scaling](#3-feature-scaling)
   - [4. Train-Test Split](#4-train-test-split)
   - [5. Logistic Regression](#5-logistic-regression)
   - [6. Random Forest Classifier](#6-random-forest-classifier)
   - [7. Model Evaluation](#7-model-evaluation)
   - [8. Model Saving](#8-model-saving)
6. [Deliverables](#deliverables)
7. [Beginner Tips](#beginner-tips)

---

## Objective

The goals of this lab are to:

1. Apply classification techniques to predict employee attrition.
2. Compare Logistic Regression and Random Forest models.
3. Understand the importance of data preprocessing and evaluation metrics.

---

## Introduction

This lab focuses on classifying whether an employee will leave the company (attrition) based on features like job role, work environment, and personal demographics. You’ll build two classification models and evaluate their performance.

---

## Dataset Overview

Dataset: **HR Employee Attrition Data**

| Column Name         | Description                                      |
|---------------------|--------------------------------------------------|
| Age                 | Age of the employee                              |
| JobRole             | Employee's job role                              |
| MaritalStatus       | Marital status of employee                       |
| MonthlyIncome       | Monthly salary                                   |
| Attrition           | Target (Yes/No for leaving the company)         |
| ...                 | Other HR and demographic features                |

---

## Python Libraries

Install the following if needed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib

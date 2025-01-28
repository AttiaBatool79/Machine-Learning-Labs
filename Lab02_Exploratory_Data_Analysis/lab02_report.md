# Lab 2: Exploratory Data Analysis (EDA)

## Table of Contents
1. [Objective](#objective)
2. [Why is EDA Important?](#why-is-eda-important)
3. [Titanic Dataset Overview](#titanic-dataset-overview)
4. [Python Libraries](#python-libraries)
5. [Steps for Exploratory Data Analysis](#steps-for-exploratory-data-analysis)
   - [1. Load the Dataset](#1-load-the-dataset)
   - [2. Analyze Columns](#2-analyze-columns)
     - [Numerical Data](#numerical-data)
     - [Categorical Data](#categorical-data)
   - [3. Relationships Between Variables](#3-relationships-between-variables)
     - [Numerical-Numerical](#numerical-numerical)
     - [Categorical-Categorical](#categorical-categorical)
     - [Numerical-Categorical](#numerical-categorical)
   - [4. Multivariate Analysis](#4-multivariate-analysis)
6. [Types of Visualizations and Their Use Cases](#types-of-visualizations-and-their-use-cases)
7. [Tasks to Complete](#tasks-to-complete)
8. [What You’ll Learn](#what-youll-learn)

---

## Objective
The goal of this lab is to perform Exploratory Data Analysis (EDA) on the Titanic dataset to:

1. Understand the dataset structure and features.
2. Perform **Univariate Analysis** to study each variable individually.
3. Conduct **Bivariate Analysis** to explore relationships between two variables.
4. Use **Multivariate Analysis** to detect patterns involving multiple variables.
5. Visualize the data effectively using different types of graphs and extract meaningful insights.

---

## Why is EDA Important?
EDA is the first step in any data analysis project. It helps:

- **Understand the data structure**: What variables exist and what they represent.
- **Detect anomalies**: Find errors, missing values, or outliers.
- **Discover relationships**: For example, how age impacts survival.
- **Prepare data**: Decide which features to include for machine learning.

---

## Titanic Dataset Overview
The Titanic dataset contains information about passengers aboard the Titanic. The key columns include:

| Column Name     | Description                                      |
|-----------------|--------------------------------------------------|
| **PassengerId** | Unique ID for each passenger.                   |
| **Survived**    | Survival status (0 = No, 1 = Yes).              |
| **Pclass**      | Ticket class (1st, 2nd, or 3rd class).          |
| **Name**        | Passenger's full name.                          |
| **Sex**         | Passenger's gender.                             |
| **Age**         | Passenger's age.                                |
| **SibSp**       | Number of siblings or spouses aboard.           |
| **Parch**       | Number of parents or children aboard.           |
| **Ticket**      | Ticket number.                                  |
| **Fare**        | Ticket fare paid.                               |
| **Cabin**       | Cabin number.                                   |
| **Embarked**    | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton). |

---

## Python Libraries
The following libraries are essential for this lab:

1. **pandas**: Data manipulation and analysis.
2. **numpy**: Numerical computations.
3. **matplotlib**: Basic data visualizations.
4. **seaborn**: Advanced statistical visualizations.

Install them using:

```bash
pip install pandas numpy matplotlib seaborn
```

---

## Steps for Exploratory Data Analysis

### 1. Load the Dataset
The Titanic dataset is loaded using the `pandas` library:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Display the first few rows
print(df.head())

# Display dataset information
print(df.info())

# Display descriptive statistics
print(df.describe())
```

- **head()**: Displays the first 5 rows for a quick overview.
- **info()**: Provides column data types and non-null counts, useful for spotting missing data.
- **describe()**: Summarizes numerical columns (mean, median, min, max, etc.).

### 2. Analyze Columns
#### **Numerical Data**
- **Histograms** show frequency distributions:

```python
sns.histplot(df['Fare'], kde=True, color='blue')
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.show()
```

- **Boxplots** detect outliers:

```python
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')
plt.show()
```

#### **Categorical Data**
- Use **countplots** for category frequencies:

```python
sns.countplot(x='Pclass', data=df, palette='viridis')
plt.title('Passenger Count by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()
```

- Use bar charts for other categories:

```python
df['Embarked'].value_counts().plot(kind='bar', color='orange')
plt.title('Embarked Port Count')
plt.xlabel('Port')
plt.ylabel('Count')
plt.show()
```

### 3. Relationships Between Variables
#### **Numerical-Numerical**
Explore relationships between two numerical columns with scatterplots:

```python
sns.scatterplot(x='Age', y='Fare', data=df)
plt.title('Scatterplot of Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()
```

#### **Categorical-Categorical**
Compare category distributions across another category:

```python
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()
```

#### **Numerical-Categorical**
Compare numerical data across categories:

```python
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Boxplot of Age by Survival Status')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()
```

### 4. Multivariate Analysis
Examine patterns involving multiple variables:

- **Pairplots**:

```python
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived', palette='coolwarm')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()
```

- **Heatmaps**:

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```

---

## Types of Visualizations and Their Use Cases

1. **Boxplot**: Summarizes data spread and identifies outliers.
2. **Histogram**: Displays frequency distribution.
3. **KDE Plot**: Shows data density and smooth distributions.
4. **Heatmap**: Visualizes correlations between variables using color.
5. **Pie Chart**: Shows proportions or percentages.
6. **Countplot**: Compares category counts.
7. **Scatterplot**: Explores relationships between two numerical variables.
8. **Pairplot**: Analyzes multiple relationships simultaneously.
9. **Bar Chart**: Visualizes category data as bars.
10. **Line Plot**: Tracks changes over time or sequences.
11. **Violin Plot**: Combines boxplot and KDE for distribution and density.
12. **Jointplot**: Combines scatter and distribution plots for two variables.
13. **Stacked Bar Chart**: Shows category proportions within groups.

---

## Tasks to Complete

### **Task 1: Dataset Exploration**
- **What to Do:**
  - Use `head()`, `info()`, and `describe()` to examine the dataset structure.
  - Identify missing values and duplicates.
- **Outcome:**
  - Gain a quick overview of the dataset structure.
  - Spot incomplete or incorrect data entries.

### **Task 2: Numerical Data Analysis**
- **What to Do:**
  - Analyze columns like `Age` and `Fare` using histograms and boxplots.
  - Detect outliers and study data spread.
- **Outcome:**
  - Understand distributions and identify potential data cleaning needs.

### **Task 3: Categorical Data Analysis**
- **What to Do:**
  - Explore columns like `Pclass`, `Sex`, and `Embarked` using `value_counts()`.
  - Visualize the results with countplots and pie charts.
- **Outcome:**
  - Understand the frequency and proportions of categories.

### **Task 4: Relationships Between Variables**
- **What to Do:**
  - Use scatterplots, boxplots, and grouped bar charts to examine relationships between variables.
- **Outcome:**
  - Identify how variables like `Age` and `Survived` or `Pclass` and `Survived` are connected.

### **Task 5: Multivariate Analysis**
- **What to Do:**
  - Use pairplots and heatmaps to examine multiple variables simultaneously.
- **Outcome:**
  - Detect overall patterns and correlations among features.

---

## What You’ll Learn
By completing this lab, you will:

1. Understand how to explore and clean a dataset.
2. Learn to use Python for creating visualizations.
3. Develop the skills to interpret data and extract insights.
4. Be better prepared for advanced analytics and machine learning projects.

# Lab 2: Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is the process of exploring and understanding a dataset. It involves looking at the data, finding patterns, and using graphs to summarize the information. EDA helps us prepare the data for further analysis or machine learning.

---

## Why is EDA Important?

EDA is important because:
- It helps us understand what the data contains.
- It shows errors, missing values, or outliers.
- It reveals relationships between variables (e.g., age and survival).
- It helps us decide which features to use for predictions.

---

# Titanic Dataset Analysis

This lab is about exploring the Titanic dataset to understand the data better and find useful patterns. It helps to know who survived, how age or ticket class affected survival, and other insights.
The dataset we are analyzing is about passengers on the Titanic. It contains the following information:
- **PassengerId**: A unique ID for each passenger.
- **Survived**: Whether the passenger survived (1 = Yes, 0 = No).
- **Pclass**: The class of the ticket (1 = First, 2 = Second, 3 = Third).
- **Name**: The passenger's full name.
- **Sex**: The passenger's gender.
- **Age**: The passenger's age.
- **SibSp**: Number of siblings or spouses onboard.
- **Parch**: Number of parents or children onboard.
- **Fare**: The ticket fare paid.
- **Cabin**: The cabin number.
- **Embarked**: The port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

---

## Tools Used

We use the following Python libraries:
1. **pandas**: To manage and analyze the data.
2. **numpy**: To do calculations with numbers.
3. **matplotlib**: To make simple graphs.
4. **seaborn**: To make beautiful and detailed graphs.

---

## Steps to Follow

1. **Load the Dataset**: Read the Titanic dataset using Python.
2. **Look at the Data**: See the first few rows, column names, and missing values.
3. **Analyze Columns**:
   - Check individual columns like Age, Fare, and Pclass.
   - Use graphs like histograms and boxplots.
4. **Find Relationships**:
   - See how variables are connected (e.g., Age and Survival).
   - Use scatterplots, heatmaps, and pairplots.

---

## Example Graphs

# Graph Plot Types for EDA

This document explains the different types of graphs used in Exploratory Data Analysis (EDA) to visualize and understand data.

---

## 1. Boxplot
A boxplot displays the distribution of numerical data and highlights outliers. It shows the minimum, first quartile (Q1), median, third quartile (Q3), and maximum values.

## 2. Histogram
A histogram shows the frequency distribution of a single numerical variable. It helps understand the shape, spread, and peaks of the data.

## 3. KDE Plot
A Kernel Density Estimate (KDE) plot shows the probability density of a variable, helping visualize the overall distribution.

## 4. Heatmap
A heatmap visualizes the correlation between numerical variables using colors. It highlights positive and negative relationships between variables.

## 5. Scatterplot
A scatterplot shows the relationship between two numerical variables using individual data points. It helps identify trends, clusters, or outliers.

## 6. Pie Chart
A pie chart represents proportions or percentages of categories as slices of a circle. It is useful for visualizing the composition of a categorical variable.

## 7. Countplot
A countplot displays the frequency of each category in a categorical variable, making it easy to compare occurrences.

## 8. Pairplot
A pairplot shows scatterplots for all numerical variable pairs in the dataset. It is useful for detecting patterns and correlations.

## 9. Line Plot
A line plot connects data points with a line, showing trends or patterns over time or sequences.

## 10. Bar Chart
A bar chart displays data for categorical variables as bars. The height of each bar represents the frequency or value.

## 11. Violin Plot
A violin plot combines a boxplot with a KDE plot to show the distribution, density, and spread of the data.

## 12. Jointplot
A jointplot combines scatterplots and histograms (or KDE plots) to show the relationship between two variables along with their individual distributions.

## 13. Stacked Bar Chart
A stacked bar chart shows proportions of categories within each group, combining total and composition information.

---

## What You Learn

By doing this project, you will:
- Understand how to explore a dataset.
- Learn to make graphs to find insights.
- See how data analysis can solve real-world problems.


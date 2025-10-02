# Minicourse: Introduction to Machine Learning with Python

## Overview

This repository contains the courseware for an introductory minicourse on Machine Learning. The content covers fundamental concepts and the practical application of models to solve two of the most common tasks in the field: **Classification** and **Regression**.

The goal is to provide a solid foundation, from data loading and analysis to model training and interpretation of their results.

## Topics Covered

### 1. Classification: Handwritten Digit Recognition

This section presents a classification model for identifying handwritten digits (0-9). This is a canonical problem that demonstrates how a model can be trained to categorize visual information.

* **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/), a dataset with 70,000 28x28 pixel grayscale images. * **Algorithm:** Logistic Regression (`LogisticRegression`), an efficient and widely used linear model for classification problems.
* **Process:**
1. Loading and exploring the dataset.
2. Splitting the data into training and test sets (`train_test_split`).
3. Training the Logistic Regression model.
4. Running predictions on the test set.
* **Performance Evaluation Metrics:**
* **Accuracy:** Percentage of correct predictions.
* **Precision:** Proportion of positive identifications that were actually correct.
* **Recall:** Proportion of true positives that were correctly identified.
* **Confusion Matrix:** Table summarizing the prediction results for each class.

### 2. Regression: Tip Prediction

This section addresses a regression problem, the objective of which is to predict a continuous numeric value. The model was trained to estimate the tip amount (`tip`) based on the total bill amount (`total_bill`) and other variables.

* **Dataset:** `Tips`, a dataset available in the Seaborn library, containing transaction records from a restaurant.
* **Algorithm:** Linear Regression, which models the relationship between a dependent variable and one or more independent variables.
* **Process:**
1. Exploratory Data Analysis (EDA) to identify correlations.
2. Data preprocessing with `One-Hot Encoding` to convert categorical variables to numeric ones.
3. Training the Linear Regression model. 4. Interpretation of the model coefficients to derive the equation of the line:
* **Equation:** `Tip = 0.93 + 0.11 × Bill`
* **Performance Evaluation Metrics:**
* **RMSE (Root Mean Squared Error):** Measures the average magnitude of prediction errors in the same unit as the target variable.
* **R² (Coefficient of Determination):** Indicates the proportion of variance in the dependent variable that is explained by the model.

## Tools and Libraries

This project uses the following tools and libraries from the Python ecosystem:

* **Scikit-learn:** For implementing Machine Learning models and evaluation metrics.
* **Pandas:** For manipulating and analyzing data structures.
* **NumPy:** For numerical computation and array operations.
* **Matplotlib & Seaborn:** For data visualization and graph generation. * **Jupyter Notebook:** As an interactive development environment.

## Running Instructions

To run the notebooks locally, follow the steps below:

1. **Clone the Repository:**
```bash
git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
cd your-repository
```

2. **Create and Activate a Virtual Environment (Recommended):**
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**
```bash
pip install scikit-learn seaborn pandas numpy matplotlib
```

4. **Start the Jupyter Server:**
```bash
jupyter notebook
```

After launching, navigate through the Jupyter interface to open notebooks (`.ipynb`) and run code cells.

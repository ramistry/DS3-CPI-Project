# DS3-CPI-Project

## 1. Overview
<br />
This project demonstrates how to use linear regression to predict the Consumer Price Index (CPI). The CPI is a key economic indicator used to measure inflation by tracking the average change over time in the prices paid by consumers for goods and services. This notebook applies a machine learning approach, specifically linear regression, to predict CPI values based on historical economic data.
<br />

## 2. Dataset

The dataset used for training and testing the linear regression model consists of various economic indicators that are relevant to the CPI. The specific columns and features of the dataset are:

  Independent Variables (Features): Economic factors like GDP growth, unemployment rate, etc.
  Dependent Variable (Target): CPI value
The dataset is divided into training and testing sets to evaluate the model's performance.

## 3. Structure
The project consists of the following files:

`CPI Prediction - Linear Regression.ipynb` : Jupyter notebook containing the entire workflow from data preprocessing to model evaluation.

`cpi_data.csv` : Directory containing the dataset used for the analysis.

## 4. Key Steps
Data Preprocessing:

`Load and clean the data.`

`Handle missing values, if any.`

`Split the data into training and testing sets.`


Model Training:

`Apply linear regression using Scikit-learn.`

`Fit the model on the training data.`

`Use the trained model to predict CPI values on the test set.`

Model Evaluation:

`Calculate metrics like Mean Squared Error (MSE) and R-squared (R²) to assess the model's performance.`

`Plot residuals to check the accuracy of the model’s predictions.`

## 5. Technologies Used and Requirements for Use
* `Python`
* `Jupyter Notebook`: Development environment.
* `Transformers (Hugging Face)`: A library providing pre-trained transformer models, including BERT.
* `Scikit-learn`: Machine learning library used for implementing linear regression.
* `Pandas`: Data manipulation library.
* `Matplotlib and Seaborn`: Libraries for plotting and data visualization.

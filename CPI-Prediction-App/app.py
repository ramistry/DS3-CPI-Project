from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import io
import base64


def load_and_clean_data(file_path):
    # read in data, fix formatting as the dataset needed to be transposed and columns renamed
    df = pd.read_csv('cpi_data.csv', header= None).transpose()
    df.columns = df.iloc[0]
    df = df[1:].set_index('variable')

    # change dtype of each column into floats from objects
    df = df.astype('float64')

    df.index=pd.to_datetime(df.index)

    # filling in null values with the mean of that column
    df = df.fillna(df['gas_value'].mean())
    
    return df

app = Flask(__name__)

df = load_and_clean_data('cpi_data.csv')

def train_model_for_category(df, category):
    X = df.index.year.values.reshape(-1, 1)
    y = df[category]
    model = LinearRegression()
    model.fit(X, y)
    return model

# Train models for each category
categories = ['all_items_value', 'apparel_value', 'energy_value', 'food_value', 'gas_value', 'medical_value', 'transportation_value']
models = {category: train_model_for_category(df, category) for category in categories}

# Function to predict CPI for a given year and category
def predict_cpi(year, category):
    model = models[category]
    return model.predict([[year]])[0]

@app.route('/')
def index():
    return render_template('index.html', categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    category = request.form['category']
    
    cpi_prediction = predict_cpi(year, category)
    cpi_prediction2 = predict_cpi(year - 1, category)
    inflation = (cpi_prediction - cpi_prediction2) / cpi_prediction2 * 100

    # Generate the plot
    plt.figure(figsize=(10, 5))
    plt.plot([year - 1, year], [cpi_prediction2, cpi_prediction], marker='o')
    plt.title(f'CPI Prediction for {category}')
    plt.xlabel('Year')
    plt.ylabel('CPI Value')
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Save the plot file and send the file path
    plot_path = f"static/plot_{year}_{category}.png"
    with open(plot_path, 'wb') as f:
        f.write(img.getbuffer())

    return render_template('result.html', category=category, year=year, cpi_prediction=cpi_prediction, inflation=inflation, plot_url=f"/{plot_path}")

if __name__ == '__main__':
    app.run(debug=True)
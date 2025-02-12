from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

import logging


# Set up logging
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)

# Load your trained pipeline
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

@app.route('/')
def home():
    return render_template('form.html')

def preprocess_input(raw_features):
    # Convert inputs to a DataFrame
    input_data = pd.DataFrame([raw_features], columns=[
        'Store', 'DayOfWeek', 'Date', 'Customers', 'Open', 'Promo',
        'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
        'CompetitionDistance', 'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
        'Promo2SinceYear', 'PromoInterval'
    ])
    
    # Ensure the 'Date' column is in datetime format
    input_data['Date'] = pd.to_datetime(input_data['Date'])
    input_data['Year'] = input_data['Date'].dt.year
    input_data['Month'] = input_data['Date'].dt.month
    input_data['Week'] = input_data['Date'].dt.isocalendar().week
    input_data['Day'] = input_data['Date'].dt.day
    input_data['DayOfYear'] = input_data['Date'].dt.dayofyear
    input_data['Weekday'] = input_data['Date'].dt.weekday  # Monday=0, Sunday=6
    input_data['Weekend'] = (input_data['Weekday'] >= 5).astype(int)  # 1 if Saturday/Sunday, else 0
    
    # Drop the original 'Date' column
    input_data = input_data.drop(columns=['Date'])
    
    return input_data

@app.route('/predict', methods=['POST'])
def predict():
    # Get raw features from the form
    raw_features = [
        request.form['Store'],
        request.form['DayOfWeek'],
        request.form['Date'],
        request.form['Customers'],
        request.form['Open'],
        request.form['Promo'],
        request.form['StateHoliday'],
        request.form['SchoolHoliday'],
        request.form['StoreType'],
        request.form['Assortment'],
        request.form['CompetitionDistance'],
        request.form['CompetitionOpenSinceMonth'],
        request.form['CompetitionOpenSinceYear'],
        request.form['Promo2'],
        request.form['Promo2SinceWeek'],
        request.form['Promo2SinceYear'],
        request.form['PromoInterval']
    ]

    logging.debug(f'Raw features: {raw_features}')

    # Preprocess the input data
    input_data = preprocess_input(raw_features)

    logging.debug(f'Processed input: {input_data}')


    # Make a prediction using the pipeline
    prediction = pipeline.predict(input_data)

    logging.debug(f'Prediction: {prediction}')

    # Prepare the result to display
    return render_template('result.html', features=raw_features, prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
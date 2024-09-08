from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests
import json
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Initialize the Flask app
app = Flask(__name__)

# Function to build, train and evaluate the neural network model
def make_neural_network():
    data = get_inrix_data()

    # Selecting features (X) and target variable (y)
    X = data[['Average Speed', 'Speed Limit']].astype(float)
    y = data['Congestion']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the neural network model
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=2))  # 2 input features
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='linear'))  # 1 output for regression

    # Compile the model (using mean squared error for regression)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=2, verbose=1)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    predictions = []
    for i in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]:
        x_new = pd.DataFrame([[65, i]], columns=['Average Speed', 'Speed Limit'])
        x_new_scaled = scaler.transform(x_new)
        y_new = model.predict(x_new_scaled)
        predictions.append((i, y_new[0][0]))
    return predictions

# Flask route to serve the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to make predictions and return results as JSON
@app.route('/predict')
def predict():
    predictions = make_neural_network()
    return jsonify(predictions) 
  
if __name__ == "__main__":
    app.run(debug=True)

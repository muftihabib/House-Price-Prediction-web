from flask import Flask, request, render_template
import pickle
import json
import numpy as np
import os

# Create a Flask web application
app = Flask(__name__)

# --- Global variables to hold the model and data columns ---
__model = None
__data_columns = None
__locations = None

# Define the project root path using this file's location
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_saved_artifacts():
    """Loads the saved model and column names from disk using absolute paths."""
    print("Loading saved artifacts... start")
    global __data_columns
    global __locations
    global __model

    # Load the column names from columns.json
    columns_path = os.path.join(PROJECT_ROOT, "columns.json")
    with open(columns_path, "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # Locations start after sqft, bath, bhk

    # Load the trained model from the pickle file
    model_path = os.path.join(PROJECT_ROOT, "bangalore_house_price_model.pickle")
    with open(model_path, "rb") as f:
        __model = pickle.load(f)
    print("Loading saved artifacts... done")

# Load artifacts at the module level so they're ready during testing/import
load_saved_artifacts()

def get_estimated_price(location, sqft, bhk, bath):
    """Takes user inputs and returns a price prediction."""
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    # Create input vector
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

# --- Define the routes for the web application ---

@app.route('/')
def home():
    return render_template('index.html', locations=__locations)

@app.route('/predict', methods=['POST'])
def predict():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    estimated_price = get_estimated_price(location, total_sqft, bhk, bath)

    return render_template(
        'index.html',
        prediction_text=f'Estimated Price: {estimated_price} Lakhs',
        locations=__locations
    )

# --- Run the application ---
if __name__ == "__main__":
    print("Starting Python Flask Server for Bangalore House Price Prediction...")
    app.run(host='0.0.0.0', port=3000, debug=True)

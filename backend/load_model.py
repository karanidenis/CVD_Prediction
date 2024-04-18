from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load the SavedModel
# loaded_model = 

app = Flask(__name__)
CORS(app)

# Load the machine learning model from the pickle file
def loaded_model():
    model = tf.keras.models.load_model('model1.h5')
    return model

# model = load_model()

def calculate_bmi(weight, height):
    bmi = weight // (height / 100) ** 2
    # data['bmi'] = bmi
    return bmi

def categorize_bmi(bmi):
    # bmi = float(data.bmi)
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


def categorize_age(age):
    # age = int(data.age)
    if age < 40:
        return 'Young'
    elif 40 <= age < 60:
        return 'Middle-aged'
    else:
        return 'Senior'


def categorize_bp(ap_lo, ap_hi):
    # ap_hi = int(data.ap_hi)
    # ap_lo = int(data.ap_lo)
    if ap_hi < 120 and ap_lo < 80:
        return 'Normal'
    elif ap_hi >= 140 or ap_lo >= 90:
        return 'Hypertension'
    else:
        return 'High-Normal'
    
def categorize_gender(data):
    if data['gender'] == "Male":
        return 2
    else:
        return 1


def preprocess_data(data):
    # Perform data preprocessing here
    # You can customize this function based on your preprocessing steps
    # Categorize BMI, age, and blood pressure
    bmi = calculate_bmi(data['weight'], data['height'])
    bmi_category = categorize_bmi(bmi)
    age_group = categorize_age(data['age'])
    bp_category = categorize_bp(data['ap_lo'], data['ap_hi'])
    gender = categorize_gender(data)
    
    # Construct the NumPy array with the preprocessed data
    preprocessed_data = np.array([[data['weight'], data['height'], data['age'], gender, data['ap_lo'], data['ap_hi'], bmi, data['gluc'], data['cholesterol'], data['smoke'], data['alco'], data['active'], bmi_category, age_group, bp_category]])
    # preprocessed_data = np.array([[data['weight'], data['height'], data['age'], gender, data['ap_lo'], data['ap_hi'], data['gluc'], data['cholesterol'], data['smoke'], data['alco'], data['active']]])
    
    # Map categorical values to numerical values
    bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
    preprocessed_data[:, 12] = np.vectorize(bmi_mapping.get)(preprocessed_data[:, 12])
    
    age_mapping = {'Young': 0, 'Middle-aged': 1, 'Senior': 2}
    preprocessed_data[:, 13] = np.vectorize(age_mapping.get)(preprocessed_data[:, 13])
    
    bp_mapping = {'Normal': 0, 'Hypertension': 1, 'High-Normal': 2}
    preprocessed_data[:, 14] = np.vectorize(bp_mapping.get)(preprocessed_data[:, 14])
    
    return preprocessed_data
    


def predict(model, data):
    # Perform predictions using the loaded model
    predictions = model.predict(data)
    label = (predictions >= 0.5).astype(int)
    return label[0][0]


# Route for handling predictions
@app.route('/predict', methods=['POST'])
def handle_prediction():
    try:
        # Load the model
        model = loaded_model()
        
        scaler = joblib.load('scaler.pkl')
        
        # Get input data from the request
        input_data = request.json
        
        # Preprocess the input data
        preprocessed_data = preprocess_data(input_data)
        
        # Scale the preprocessed data
        scaled_data = scaler.transform(preprocessed_data)
        
        # Make predictions
        prediction = predict(model, scaled_data)
        
        # Return the prediction result
        return jsonify({'prediction': prediction.tolist()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for confirming user input data
@app.route('/confirm-input', methods=['POST'])
def confirm_input():
    try:
        input_data = request.json
        # Check if input data exists
        if input_data:
            # If input data exists, return success message
            return jsonify({'message': 'Input data received successfully!'}), 200
        else:
            # If no input data, return error message
            return jsonify({'error': 'No input data received!'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8080)

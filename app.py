from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

# Load and preprocess data
def load_data():
    df = pd.read_csv('water_pollution_disease.csv')
    
    # Handle missing values
    mode_value = df['Water Treatment Method'].mode()[0]
    df['Water Treatment Method'].fillna(mode_value, inplace=True)
    
    # Winsorize Lead Concentration
    from scipy.stats.mstats import winsorize
    df['Lead_winsorized'] = winsorize(df['Lead Concentration (µg/L)'], limits=[0.05, 0.05])
    
    # Create binary target (1 for contaminated, 0 for not contaminated)
    # Assuming Contaminant Level > 5 ppm is contaminated (adjust threshold as needed)
    df['Contaminated'] = (df['Contaminant Level (ppm)'] > 5).astype(int)
    
    # Define feature categories
    water_metrics = ['pH Level', 'Turbidity (NTU)', 'Dissolved Oxygen (mg/L)', 
                    'Nitrate Level (mg/L)', 'Bacteria Count (CFU/mL)', 'Lead_winsorized']
    
    environmental_factors = ['Rainfall (mm per year)', 'Temperature (°C)']
    
    infrastructure_socio = ['Water Treatment Method', 'Access to Clean Water (% of Population)',
                          'Sanitation Coverage (% of Population)', 'Population Density (people per km²)']
    
    # Combine all features
    features = water_metrics + environmental_factors + infrastructure_socio
    
    # Convert categorical variables
    df = pd.get_dummies(df, columns=['Water Treatment Method', 'Country', 'Region', 'Water Source Type'])
    
    # Update features list with dummy variables
    features = [f for f in features if f not in ['Water Treatment Method']] + \
               [col for col in df.columns if 'Water Treatment Method_' in col or 
                'Country_' in col or 'Region_' in col or 'Water Source Type_' in col]
    
    X = df[features]
    y = df['Contaminated']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = water_metrics + environmental_factors + ['Access to Clean Water (% of Population)',
                                                                'Sanitation Coverage (% of Population)',
                                                                'Population Density (people per km²)']
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, scaler, features

# Load model and scaler
if not os.path.exists('model.joblib'):
    model, scaler, features = load_data()
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(features, 'features.joblib')
else:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    features = joblib.load('features.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Create DataFrame from form data
        input_data = pd.DataFrame([data])
        
        # Add dummy variables (set all to 0 initially)
        for feature in features:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Set the appropriate dummy variable to 1
        for col in input_data.columns:
            if 'Water Treatment Method_' in col or 'Country_' in col or 'Region_' in col or 'Water Source Type_' in col:
                if col.split('_')[1] == data.get(col.split('_')[0], ''):
                    input_data[col] = 1
        
        # Scale numerical features
        numerical_features = ['pH Level', 'Turbidity (NTU)', 'Dissolved Oxygen (mg/L)', 
                            'Nitrate Level (mg/L)', 'Bacteria Count (CFU/mL)', 'Lead_winsorized',
                            'Rainfall (mm per year)', 'Temperature (°C)', 
                            'Access to Clean Water (% of Population)',
                            'Sanitation Coverage (% of Population)', 
                            'Population Density (people per km²)']
        
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])
        
        # Ensure columns are in correct order
        input_data = input_data[features]
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]  # Probability of being contaminated
        
        result = {
            'prediction': 'Contaminated' if prediction[0] == 1 else 'Not Contaminated',
            'probability': float(probability),
            'confidence': 'High' if probability > 0.7 or probability < 0.3 else 'Medium' if probability > 0.6 or probability < 0.4 else 'Low'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
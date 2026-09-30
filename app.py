from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# 1. Initialize Flask server
app = Flask(__name__)

# Enable CORS so your React application (running on another port) can send requests to this server
CORS(app)

# 2. Load the trained machine learning model
try:
    model = joblib.load("logestic.pkl")
    print("LOGISTIC REGRESSION MODEL LOADED SUCCESSFULLY!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 3. Create the endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Machine learning model is not loaded.'}), 500
        
    try:
        # Get JSON data sent from React
        data = request.json
        
        # Extract study hours from request body
        if 'study_hours' not in data:
            return jsonify({'error': 'Missing study_hours parameter'}), 400
            
        hours = float(data['study_hours'])
        
        # Format input data as 2D array for scikit-learn model: [[hours]]
        input_data = np.array([[hours]])
        
        # Predict class (0 = Fail, 1 = Pass)
        prediction_val = int(model.predict(input_data)[0])
        
        # Convert prediction value strictly to 'Pass' or 'Fail'
        result = 'Pass' if prediction_val == 1 else 'Fail'
        
        # Return response in JSON format
        return jsonify({'prediction': result})
        
    except ValueError:
        return jsonify({'error': 'Invalid input value. Must be a decimal number.'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Start the Flask development server on port 5000
    print("Flask server running on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)

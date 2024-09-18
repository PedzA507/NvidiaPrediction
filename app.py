from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("Nvidia.pkl")

@app.route('/api/nvidia', methods=['POST'])
def nvidia():
    
    open = float(request.form.get('open'))
    high = float(request.form.get('high'))
    low = float(request.form.get('low'))
    close = float(request.form.get('close'))  # Removed 'close' from the input
    adjclose = float(request.form.get('adjclose'))
    volume = float(request.form.get('volume'))  # Removed 'close' from the input
    
    # Prepare the input for the model
    x = np.array([[open, high, low, adjclose, volume]])  # Use 5 features

    # Predict using the model
    prediction = model.predict(x)

    # Return the result
    return {'price': round(prediction[0], 2)}, 200    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)

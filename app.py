from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

#Loading the pre-trained model using joblib
model = joblib.load('model/churn_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all form values with default values if not provided
        credit_score = float(request.form.get('CreditScore', 0))
        gender = request.form.get('Gender', 'Male')
        age = float(request.form.get('age', 0))
        balance = float(request.form.get('Balance', 0))
        num_products = float(request.form.get('NumOfProducts', 0))
        is_active_member = 1 if request.form.get('is_active_member') == '1' else 0
        geography = request.form.get('Geography', 'France')
        
        # Set geography one-hot encoding
        is_france = 1 if geography == 'France' else 0
        is_germany = 1 if geography == 'Germany' else 0
        is_spain = 1 if geography == 'Spain' else 0

        # Create feature array
        input_features = [
            credit_score,
            1 if gender == 'Male' else 0,
            age,
            0,  # Tenure (since it's not in the form)
            balance,
            num_products,
            is_active_member,
            is_france,
            is_germany,
            is_spain
        ]

        # Reshape and make prediction
        input_data = np.array(input_features).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Return results
        return render_template('results.html',
                           prediction=prediction,
                           probability=probability*100)
                           
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # Add this for debugging
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
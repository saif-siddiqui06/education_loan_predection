from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Global variables for model and scaler
model = None
scaler = None

# Define the feature names (Must match training phase)
feature_names = ['income', 'loan_amount', 'interest_rate', 'previous_loan_status', 'debt_to_income_ratio']

def load_model():
    """Load the machine learning model and scaler once when the app starts."""
    global model, scaler
    try:
        with open('loan_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        print("✅ Model and scaler loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise Exception("Failed to load the model or scaler.")

# Load the model on startup
load_model()

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process form submission, make a loan prediction, and return the result."""
    try:
        # Debugging: Log received form data
        print("Received Form Data:", request.form)

        # Ensure all required fields exist
        required_fields = ['income', 'loan_amount', 'interest_rate', 'previous_loan_status']
        if not all(k in request.form for k in required_fields):
            return render_template('index.html', error="Error: Missing form data!"), 400

        # Get user input from form
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        interest_rate = float(request.form['interest_rate'])
        previous_loan_status = int(request.form['previous_loan_status'])

        # Compute derived feature (debt_to_income_ratio) safely
        debt_to_income_ratio = loan_amount / income if income > 0 else 0

        # Prepare data in the correct order
        user_input = [[income, loan_amount, interest_rate, previous_loan_status, debt_to_income_ratio]]
        
        # Convert to DataFrame to maintain feature names
        user_df = pd.DataFrame(user_input, columns=feature_names)

        # Debugging: Log input DataFrame
        print("User DataFrame:\n", user_df)

        # Scale input
        user_scaled = scaler.transform(user_df)

        # Make prediction
        prediction = model.predict(user_scaled)[0]
        probability = model.predict_proba(user_scaled)[0][1] * 100  # Probability of approval

        # Determine approval message
        approval_text = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"

        return render_template('index.html', approval=approval_text, probability=round(probability, 2))

    except Exception as e:
        logging.error(f"Prediction Error: {e}")  # Log the error
        return render_template('index.html', error="An error occurred while processing your request. Please try again."), 400

@app.route('/predict-api', methods=['POST'])
def predict_api():
    """API version of the prediction function returning JSON response."""
    try:
        data = request.get_json()

        required_fields = ['income', 'loan_amount', 'interest_rate', 'previous_loan_status']
        if not all(k in data for k in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        income = float(data['income'])
        loan_amount = float(data['loan_amount'])
        interest_rate = float(data['interest_rate'])
        previous_loan_status = int(data['previous_loan_status'])

        debt_to_income_ratio = loan_amount / income if income > 0 else 0

        user_input = [[income, loan_amount, interest_rate, previous_loan_status, debt_to_income_ratio]]
        user_df = pd.DataFrame(user_input, columns=feature_names)

        user_scaled = scaler.transform(user_df)

        prediction = model.predict(user_scaled)[0]
        probability = model.predict_proba(user_scaled)[0][1] * 100

        approval_text = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"

        return jsonify({
            "approval": approval_text,
            "probability": round(probability, 2)
        })

    except Exception as e:
        logging.error(f"Prediction API Error: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again."}), 400

@app.route('/health')
def health():
    """Health check route to verify if the server is running."""
    return "Server is Running!", 200

if __name__ == '__main__':
    app.run(debug=True)

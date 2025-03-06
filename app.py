from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import logging


app = Flask(__name__)

def handler(event, context):
    return app(event, context)



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
    try:
        # Get user input
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        interest_rate = float(request.form['interest_rate'])
        previous_loan_status = int(request.form['previous_loan_status'])
        loan_duration = int(request.form['loan_duration'])

        # Compute derived features
        debt_to_income_ratio = loan_amount / income

        # Prepare data for model prediction
        user_input = [[income, loan_amount, interest_rate, previous_loan_status, debt_to_income_ratio]]
        user_df = pd.DataFrame(user_input, columns=feature_names)

        # Scale input and make prediction
        user_scaled = scaler.transform(user_df)
        prediction = model.predict(user_scaled)[0]
        probability = model.predict_proba(user_scaled)[0][1] * 100  # Approval probability

        # Loan calculations
        monthly_rate = (interest_rate / 100) / 12
        months = loan_duration * 12
        emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
        total_amount_paid = emi * months
        total_interest_paid = total_amount_paid - loan_amount

        # Loan risk indicator
        risk_level = "Low" if debt_to_income_ratio < 0.3 else "Medium" if debt_to_income_ratio < 0.6 else "High"

        # Dynamic Advice Based on Probability
        if probability > 75:
            advice = "✅ Your chances of approval are great! Consider checking different banks for better rates."
        elif probability > 50:
            advice = "⚠️ Your approval chances are moderate. You may improve your credit score or reduce loan amount."
        else:
            advice = "❌ Approval chances are low. Consider applying for a lower loan amount or adding a co-applicant."

        # Approval message
        approval_text = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"

        return render_template('index.html', 
                               approval=approval_text, 
                               probability=round(probability, 2),
                               emi=round(emi, 2),
                               total_amount_paid=round(total_amount_paid, 2),
                               total_interest_paid=round(total_interest_paid, 2),
                               risk_level=risk_level,
                               advice=advice)

    except Exception as e:
        return render_template('index.html', error=f"Error: {e}"), 400



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

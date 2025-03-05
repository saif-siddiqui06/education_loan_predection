from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open('loan_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the feature names (Must match training phase)
feature_names = ['income', 'loan_amount', 'interest_rate', 'previous_loan_status', 'debt_to_income_ratio']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debugging: Check if form data is received
        print("Received Form Data:", request.form)

        # Ensure all form inputs exist
        required_fields = ['income', 'loan_amount', 'interest_rate', 'previous_loan_status']
        if not all(k in request.form for k in required_fields):
            return render_template('index.html', error="Error: Missing form data!"), 400

        # Get user input from form
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        interest_rate = float(request.form['interest_rate'])
        previous_loan_status = int(request.form['previous_loan_status'])  # Corrected name

        # Compute derived feature (debt_to_income_ratio)
        debt_to_income_ratio = loan_amount / income

        # Prepare data in the correct order
        user_input = [[income, loan_amount, interest_rate, previous_loan_status, debt_to_income_ratio]]
        
        # Convert to DataFrame to maintain feature names
        user_df = pd.DataFrame(user_input, columns=feature_names)

        # Debugging: Check input format before scaling
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
        return render_template('index.html', error=f"Error: {e}"), 400

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, render_template
import joblib
print("JOBLIB IMPORTED SUCCESSFULLY")
import pandas as pd

app = Flask(__name__)

# Load model and features
model = joblib.load("loan_model.pkl")
features = joblib.load("features.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # Convert values to float
        for key in data:
            data[key] = float(data[key])

        df = pd.DataFrame([data])

        # Feature engineering
        df["loan_income_ratio"] = df["loan_amount"] / df["monthly_income"]
        df["balance_income_ratio"] = df["account_balance"] / df["monthly_income"]

        df = df[features]

        probabilities = model.predict_proba(df)[0]

        default_prob = probabilities[0]
        repay_prob = probabilities[1]

        if repay_prob >= 0.65:
            decision = "APPROVE"
        elif repay_prob >= 0.50:
            decision = "REVIEW / REDUCE LOAN"
        else:
            decision = "REJECT"

        return render_template(
            "index.html",
            prediction_text=f"Repayment Probability: {round(repay_prob*100,2)}% | Decision: {decision}"
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
    
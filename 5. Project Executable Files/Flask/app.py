from flask import Flask, render_template, request
import pickle
import numpy as np

# Load models and scalers
try:
    ms = pickle.load(open("bestmodel_2.pk1", "rb"))  # MinMaxScaler or StandardScaler
    model = pickle.load(open("rf_model_2.pk1", "rb"))  # RandomForest Model
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    exit(1)

app = Flask(__name__)

@app.route('/')
def loadpage():
    return render_template("index.html")

@app.route('/y_predict', methods=["POST"])
def prediction():
    try:
        # Get form data
        Cost_of_the_Product = request.form["Cost_of_the_Product"]
        Discount_offered = request.form["Discount_offered"]
        Prior_purchases = request.form["Prior_purchases"]
        Weight_in_gms = request.form["Weight_in_gms"]
        Product_importance_low = request.form.get("Product_importance_low") == 'True'
        Product_importance_medium = request.form.get("Product_importance_medium") == 'True'
        Customer_rating = request.form["Customer_rating"]
        Customer_care_calls = request.form["Customer_care_calls"]

        # Prepare input data
        preds = [[
            float(Cost_of_the_Product), 
            float(Customer_rating), 
            int(Customer_care_calls), 
            int(Prior_purchases), 
            float(Product_importance_low), 
            float(Product_importance_medium), 
            float(Discount_offered), 
            float(Weight_in_gms)
        ]]

        # Transform and predict
        transformed_preds = ms.transform(preds)
        prediction = model.predict(transformed_preds)
        prediction_proba = model.predict_proba(transformed_preds)[0]

        not_reach_prob = prediction_proba[0]
        reach_prob = prediction_proba[1]

        prediction_text = 'There is a {:.2f}% chance that your product will reach in time'.format(reach_prob * 100)
        print(prediction_text)
        print(prediction)

        return render_template("index.html", prediction_text=prediction_text)
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == "__main__":
    app.run(debug=False)

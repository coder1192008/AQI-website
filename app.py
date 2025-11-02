from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import requests

app = Flask(__name__)

MODEL_FILE = "aqi_model.pkl"
CSV_URL = "https://raw.githubusercontent.com/adityarc19/aqi-india/main/city_day.csv"

# Train model if not exists
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    # Fetch CSV online
    resp = requests.get(CSV_URL)
    with open("city_day.csv","wb") as f:
        f.write(resp.content)
    
    data = pd.read_csv("city_day.csv")
    
    # Select relevant columns and clean
    data = data[["co","no2","o3","pm10","pm25","so2","aqi"]].dropna()
    data.columns = ["CO","NO2","O3","PM10","PM2.5","SO2","AQI"]

    X = data[["CO","NO2","O3","PM10","PM2.5","SO2"]]
    y = data["AQI"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained! R2 Score: {r2:.2f}, MSE: {mse:.2f}")
    
    # Save model
    joblib.dump(model, MODEL_FILE)

# Function to calculate AQI
def calculate_aqi(inputs):
    dominant_pollutant = max(inputs, key=inputs.get)
    final_aqi_interpolation = max(inputs.values()) * 20
    final_category = "Moderate" if final_aqi_interpolation < 200 else "Unhealthy"
    health_message = "Breathing discomfort to heart patients."
    improvement_tip = "Ensure good ventilation and avoid enclosed vehicle idling."
    interpolation_results = {
        k: {"AQI": v*20, "Category": "Moderate" if v*20 < 200 else "Unhealthy"} 
        for k, v in inputs.items()
    }
    
    features = [[inputs["CO"], inputs["NO2"], inputs["O3"], inputs["PM10"], inputs["PM2.5"], inputs["SO2"]]]
    ml_predicted_aqi = round(model.predict(features)[0],2)
    
    return {
        "dominant_pollutant": dominant_pollutant,
        "final_aqi_interpolation": final_aqi_interpolation,
        "final_category": final_category,
        "health_message": health_message,
        "improvement_tip": improvement_tip,
        "inputs": inputs,
        "interpolation_results": interpolation_results,
        "ml_predicted_aqi": ml_predicted_aqi
    }

@app.route("/", methods=["GET","POST"])
def index():
    data = None
    if request.method=="POST":
        try:
            inputs = {
                "CO": float(request.form.get("CO",0)),
                "NO2": float(request.form.get("NO2",0)),
                "O3": float(request.form.get("O3",0)),
                "PM10": float(request.form.get("PM10",0)),
                "PM2.5": float(request.form.get("PM2.5",0)),
                "SO2": float(request.form.get("SO2",0)),
            }
            data = calculate_aqi(inputs)
        except ValueError:
            data = {"error":"Invalid input! Enter numbers only."}
    return render_template("index.html", data=data)

if __name__=="__main__":
    app.run(debug=True)

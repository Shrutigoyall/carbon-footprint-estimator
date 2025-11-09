# predict.py
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/model.joblib')

def prepare_input(electricity_kwh, vehicle_km, vehicle_type, flights_hours, food_type, waste_kg):
    """Prepare data in the same format as training."""
    df = pd.DataFrame([{
        'electricity_kwh': electricity_kwh,
        'vehicle_km': vehicle_km,
        'flights_hours': flights_hours,
        'waste_kg': waste_kg,
        'vehicle_type_cat': {'petrol':0,'diesel':1,'ev':2,'public':3}[vehicle_type],
        'food_score': {'meat_heavy':2,'mixed':1,'veg':0}[food_type]
    }])
    return df

def predict_co2(electricity_kwh, vehicle_km, vehicle_type, flights_hours, food_type, waste_kg):
    """Return predicted CO2 emission in kg."""
    X = prepare_input(electricity_kwh, vehicle_km, vehicle_type, flights_hours, food_type, waste_kg)
    predicted = model.predict(X)[0]
    return float(predicted)

def recommend_actions(electricity_kwh, vehicle_km, vehicle_type, flights_hours, food_type, waste_kg, predicted):
    """Give simple recommendations to reduce emissions."""
    recs = []
    if electricity_kwh > 250:
        recs.append("Reduce electricity use — switch to energy-efficient appliances or solar power.")
    if vehicle_type in ['petrol', 'diesel'] and vehicle_km > 500:
        recs.append("Consider using public transport, carpooling, or EVs for long distances.")
    if flights_hours > 5:
        recs.append("Limit frequent air travel; prefer trains or online meetings where possible.")
    if food_type == 'meat_heavy':
        recs.append("Reduce red meat consumption; try plant-based meals a few days a week.")
    if waste_kg > 15:
        recs.append("Segregate and recycle waste to lower your carbon impact.")
    if not recs:
        recs.append("Great job! Your carbon footprint is already low — keep it up!")
    return recs

# Test (optional)
if __name__ == "__main__":
    result = predict_co2(250, 800, "petrol", 2, "mixed", 10)
    print(f"Predicted CO2: {result:.2f} kg/month")
    tips = recommend_actions(250, 800, "petrol", 2, "mixed", 10, result)
    print("Recommendations:")
    for t in tips:
        print("-", t)

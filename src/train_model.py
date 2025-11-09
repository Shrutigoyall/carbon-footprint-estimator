# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Simulate dataset (for now)
n = 1000
rng = np.random.default_rng(42)
df = pd.DataFrame({
    'electricity_kwh': rng.normal(200,50,n).clip(20),
    'vehicle_km': rng.normal(800,300,n).clip(0),
    'vehicle_type': rng.choice(['petrol','diesel','ev','public'], n, p=[0.5,0.2,0.05,0.25]),
    'flights_hours': rng.exponential(1.0, n),
    'food_type': rng.choice(['meat_heavy','mixed','veg'], n, p=[0.3,0.5,0.2]),
    'waste_kg': rng.normal(10,3,n).clip(0)
})
efs = {'electricity_kwh':0.9, 'vehicle_km':0.21, 'flights_hours':90, 'waste_kg':2}
df['total_co2_kg'] = (
    df['electricity_kwh']*efs['electricity_kwh'] +
    df['vehicle_km']*efs['vehicle_km'] +
    df['flights_hours']*efs['flights_hours'] +
    df['waste_kg']*efs['waste_kg'] +
    np.where(df['food_type']=='meat_heavy', 400, np.where(df['food_type']=='mixed', 250, 150))
)

# Encode categorical columns
df['vehicle_type_cat'] = df['vehicle_type'].map({'petrol':0,'diesel':1,'ev':2,'public':3})
df['food_score'] = df['food_type'].map({'meat_heavy':2,'mixed':1,'veg':0})

X = df[['electricity_kwh','vehicle_km','flights_hours','waste_kg','vehicle_type_cat','food_score']]
y = df['total_co2_kg']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"✅ Model trained successfully!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.3f}")

# Save the model
import os
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.joblib')
print("Model saved to models/model.joblib")

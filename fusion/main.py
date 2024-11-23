import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime

# Load data
df = pd.read_csv("traffic.csv")

# Preprocess data
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df = df.drop(columns=['ID', 'DateTime'])

# Define features and target
X = df[['Junction', 'Hour', 'Day', 'Month', 'DayOfWeek']]
y = df['Vehicles']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if the model is already saved
try:
    # Load model if it exists
    with open("traffic_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded from file.")
except FileNotFoundError:
    # Train and save the model if it doesn't exist
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    with open("traffic_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved to file.")

# Evaluate model
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Predict for a specific date
def predict_traffic(date, junction):
    dt = pd.to_datetime(date)
    input_features = {
        'Junction': [junction],
        'Hour': [dt.hour],
        'Day': [dt.day],
        'Month': [dt.month],
        'DayOfWeek': [dt.dayofweek]
    }
    input_df = pd.DataFrame(input_features)
    prediction = model.predict(input_df)
    return prediction[0]

# Example prediction
print("Predicted Vehicles:", predict_traffic("2023-11-08 08:00:00", 1))

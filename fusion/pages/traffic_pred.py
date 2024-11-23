import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load
import os

# Load the traffic data
df = pd.read_csv("traffic.csv")

# Preprocess the data
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df = df.drop(columns=['ID', 'DateTime'])

# Define features and target
X = df[['Junction', 'Hour', 'Day', 'Month', 'DayOfWeek']]
y = df['Vehicles']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model path
model_path = "traffic_model.joblib"

# Load or train the model
if os.path.exists(model_path):
    # Load the model if it exists
    model = load(model_path)
    st.write("Model loaded from file.")
else:
    # Train the model and save it if it doesn't exist
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    dump(model, model_path)
    st.write("Model trained and saved to file.")

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# st.write("Mean Absolute Error:", mae)
# st.write("Mean Squared Error:", mse)

# Define a function to predict traffic
def predict_traffic(date_time, junction):
    dt = pd.to_datetime(date_time)
    input_features = {
        'Junction': [junction],
        'Hour': [dt.hour],
        'Day': [dt.day],
        'Month': [dt.month],
        'DayOfWeek': [dt.dayofweek]
    }
    input_df = pd.DataFrame(input_features)
    prediction = model.predict(input_df)
    return int(round(prediction[0]))

# Initialize default date and time in session state
if 'default_date' not in st.session_state:
    st.session_state['default_date'] = datetime.now().date()
if 'default_time' not in st.session_state:
    st.session_state['default_time'] = datetime.now().time()

# Streamlit app layout
st.title("Traffic Prediction App")



# Editable date and time fields (without direct session_state update after widget creation)
selected_date = st.date_input("Select a date", value=st.session_state['default_date'])
selected_time = st.time_input("Select a time", value=st.session_state['default_time'])
junction = st.selectbox("Select a Junction", sorted(df['Junction'].unique()))

# Prediction period input
prediction_period = st.number_input("Enter prediction duration (in hours)", min_value=1, value=5)

# Predict button
if st.button("Predict Traffic"):
    # Combine selected date and time for prediction start
    start_datetime = datetime.combine(selected_date, selected_time)
    end_datetime = start_datetime + timedelta(hours=prediction_period)

    st.write(f"Predicted traffic for {prediction_period} hours starting from {start_datetime}:")
    
    predictions = []
    prediction_times = []
    
    # Loop through each hour in the prediction period
    current_time = start_datetime
    while current_time < end_datetime:
        prediction = predict_traffic(current_time, junction)
        predictions.append(prediction)
        prediction_times.append(current_time)
        current_time += timedelta(hours=1)
    
    # Display results
    results_df = pd.DataFrame({"Time": prediction_times, "Predicted Vehicles": predictions})
    st.write(results_df)
    
    st.line_chart(results_df.set_index("Time"))

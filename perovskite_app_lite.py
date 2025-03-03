
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Title and intro
st.title("Perovskite Automated Synthesis System - Lite Version")
st.write("This app predicts the optimal **Bandgap** for perovskite ink formulations using a trained Random Forest model.")

# Load data
@st.cache_data
def load_data():
    data = pd.read_excel("PL VALUES.xlsx")
    data['Bandgap'] = 1240 / data['Wavelength'].replace(0, np.nan)
    data = data.dropna(subset=['Bandgap'])
    return data

data = load_data()

# Display raw data (optional)
if st.checkbox("Show Raw Data"):
    st.write(data)

# Preprocessing (Encode categorical variables)
data_encoded = pd.get_dummies(data, columns=["Ink", "Additive"], drop_first=True)

# Features and target
X = data_encoded[['Intensity'] + [col for col in data_encoded.columns if "Ink_" in col or "Additive_" in col]]
y = data_encoded['Bandgap']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Model Performance:** MAE = {mae:.4f}, R² = {r2:.4f}")

# Feature Importance Plot
importance = model.feature_importances_
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)

st.write("### Feature Importance")
st.bar_chart(importance_df.set_index("Feature"))

# User input for prediction
st.write("## Predict Bandgap for New Conditions")
intensity = st.number_input("Enter Intensity:", min_value=0, value=5000)

# Dynamic dropdowns based on dataset values
ink_options = data['Ink'].unique().tolist()
additive_options = data['Additive'].unique().tolist()

ink = st.selectbox("Select Ink Type:", ink_options)
additive = st.selectbox("Select Additive Type:", additive_options)

# Prepare input for prediction
input_data = pd.DataFrame([[intensity]], columns=['Intensity'])

# Add categorical columns (set all to 0 initially)
for col in X.columns:
    if "Ink_" in col or "Additive_" in col:
        input_data[col] = 0

# Set the right ink and additive columns to 1 based on user selection
ink_col = f"Ink_{ink}"
additive_col = f"Additive_{additive}"

if ink_col in input_data.columns:
    input_data[ink_col] = 1
if additive_col in input_data.columns:
    input_data[additive_col] = 1

# Predict Bandgap
predicted_bandgap = model.predict(input_data)[0]

st.write(f"### Predicted Bandgap: **{predicted_bandgap:.4f} eV**")

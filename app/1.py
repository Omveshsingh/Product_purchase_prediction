import streamlit as st
import pickle
import numpy as np
import os

st.title("Product Purchase Predictor")

# Model selection
model_options = {
    "Decision Tree": "decision_tree_model.pkl",
    "Logistic Regression": "logistic_model.pkl",
    "Random Forest Classifier": "random_forest_model.pkl"
}

selected_model = st.selectbox("Choose Model", list(model_options.keys()))

# Load selected model
model_path = os.path.join("D:\product_purchase_prediction\model", model_options[selected_model])
model = pickle.load(open(model_path, "rb"))

# Input fields
st.subheader("Customer Information")
time = st.number_input("Time Spent on Website", min_value=0.0, max_value=60.0, value=10.0, step=0.1, format="%.1f")
age = st.number_input("Age", 10, 100)
gender = st.radio("Gender", ['Male', 'Female'])
ads = st.slider("Ads Clicked", 0, 20)
purchases = st.slider("Previous Purchases", 0, 30)

# Preprocess input
gender = 1 if gender == 'Male' else 0
input_data = np.array([[time, age, gender, ads, purchases]])

# Prediction
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)
    st.success("Will Buy!" if prediction[0] == 1 else "Will Not Buy.")
    probability = model.predict_proba(input_data)[0][1]
    st.write(f"📈 **Confidence that customer will buy:** {round(probability * 100, 2)}%")
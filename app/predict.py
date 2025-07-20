import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

# Set wide layout
st.set_page_config(layout="wide")
st.title("🛒 Product Purchase Prediction: Model Comparison")

# Load models
base_path = os.path.dirname(__file__)
logistic_model = pickle.load(open(os.path.join(base_path, "logistic_model.pkl"), "rb"))
tree_model = pickle.load(open(os.path.join(base_path, "decision_tree_model.pkl"), "rb"))

# Sidebar inputs
st.sidebar.header("Enter Customer Info")
time_spent = st.sidebar.number_input("Time Spent on Website (minutes)", min_value=0.0, max_value=60.0, value=10.0, step=0.1)
age = st.sidebar.slider("Age", min_value=10, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
ads_clicked = st.sidebar.slider("Number of Ads Clicked", 0, 20, 5)
previous_purchases = st.sidebar.slider("Previous Purchases", 0, 30, 3)

# Encode gender
gender_encoded = 1 if gender == "Male" else 0

# Prepare input data
input_data = np.array([[time_spent, age, gender_encoded, ads_clicked, previous_purchases]])

# Predict and display results
if st.sidebar.button("Predict Purchase"):
    col1, col2 = st.columns(2)

    models = {
        "Logistic Regression": (logistic_model, col1, 'royalblue'),
        "Decision Tree": (tree_model, col2, 'seagreen')
    }

    for name, (model, column, color) in models.items():
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        with column:
            st.markdown(f"### {name} Model")
            st.markdown("**Purchase Prediction:** " + 
            ('<span style="color:green">Yes</span>' if prediction == 1 else '<span style="color:red">No</span>'),
            unsafe_allow_html=True)

            st.markdown("**Purchase Probability:**")
            st.progress(probability)

            # Donut Chart
            fig, ax = plt.subplots()
            labels = ['No Purchase', 'Purchase']
            sizes = [1 - probability, probability]
            colors = ['lightgray', color]
            ax.pie(sizes, labels=labels, colors=colors, startangle=90, wedgeprops=dict(width=0.4))
            ax.set_title(f"{name} Prediction")
            st.pyplot(fig)

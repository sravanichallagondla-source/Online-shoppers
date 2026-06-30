import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model
model = load_model("online_shoppers_model.keras")

# Load preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

st.set_page_config(page_title="Online Shopping Intention Prediction")

st.title("🛒 Online Shopping Intention Prediction")

st.write("Enter customer details below:")

# Numerical Inputs
Administrative = st.number_input("Administrative", min_value=0, value=2)
Administrative_Duration = st.number_input("Administrative Duration", value=50.0)

Informational = st.number_input("Informational", min_value=0, value=0)
Informational_Duration = st.number_input("Informational Duration", value=0.0)

ProductRelated = st.number_input("Product Related", min_value=0, value=30)
ProductRelated_Duration = st.number_input("Product Related Duration", value=800.0)

BounceRates = st.number_input("Bounce Rates", value=0.02, format="%.5f")
ExitRates = st.number_input("Exit Rates", value=0.04, format="%.5f")
PageValues = st.number_input("Page Values", value=0.0)
SpecialDay = st.number_input("Special Day", value=0.0)

Month = st.selectbox(
    "Month",
    ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
)

OperatingSystems = st.number_input("Operating Systems", min_value=1, value=2)
Browser = st.number_input("Browser", min_value=1, value=2)
Region = st.number_input("Region", min_value=1, value=1)
TrafficType = st.number_input("Traffic Type", min_value=1, value=2)

VisitorType = st.selectbox(
    "Visitor Type",
    ['Returning_Visitor', 'New_Visitor', 'Other']
)

Weekend = st.selectbox("Weekend", [False, True])

if st.button("Predict"):

    data = {
        "Administrative": Administrative,
        "Administrative_Duration": Administrative_Duration,
        "Informational": Informational,
        "Informational_Duration": Informational_Duration,
        "ProductRelated": ProductRelated,
        "ProductRelated_Duration": ProductRelated_Duration,
        "BounceRates": BounceRates,
        "ExitRates": ExitRates,
        "PageValues": PageValues,
        "SpecialDay": SpecialDay,
        "Month": Month,
        "OperatingSystems": OperatingSystems,
        "Browser": Browser,
        "Region": Region,
        "TrafficType": TrafficType,
        "VisitorType": VisitorType,
        "Weekend": Weekend
    }

    import pandas as pd

    input_df = pd.DataFrame([data])

    input_processed = preprocessor.transform(input_df)

    prediction = model.predict(input_processed)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        st.success("✅ Customer is likely to Purchase.")
    else:
        st.error("❌ Customer is unlikely to Purchase.")

    st.write(f"**Purchase Probability:** {probability:.2%}")
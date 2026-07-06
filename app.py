import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
 
# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Online Shopping Intention Prediction",
    page_icon="🛒",
    layout="centered"
)
 
# -------------------------------
# Load Model
# -------------------------------
model = load_model("online_shoppers_model.keras")
 
# -------------------------------
# Load Preprocessor
# -------------------------------
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)
 
# -------------------------------
# Title
# -------------------------------
st.title("🛒 Online Shopping Intention Prediction")
st.write("Enter the customer details below.")
 
# -------------------------------
# Input Fields
# -------------------------------
Administrative = st.number_input("Administrative", min_value=0, value=0)
Administrative_Duration = st.number_input("Administrative Duration", value=0.0)
Informational = st.number_input("Informational", min_value=0, value=0)
Informational_Duration = st.number_input("Informational Duration", value=0.0)
ProductRelated = st.number_input("Product Related", min_value=0, value=1)
ProductRelated_Duration = st.number_input("Product Related Duration", value=0.0)
BounceRates = st.number_input("Bounce Rates", value=0.0)
ExitRates = st.number_input("Exit Rates", value=0.0)
PageValues = st.number_input("Page Values", value=0.0)
SpecialDay = st.number_input("Special Day", value=0.0)
Month = st.selectbox(
    "Month",
    ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
)
OperatingSystems = st.number_input("Operating Systems", min_value=1, value=1)
Browser = st.number_input("Browser", min_value=1, value=1)
Region = st.number_input("Region", min_value=1, value=1)
TrafficType = st.number_input("Traffic Type", min_value=1, value=1)
VisitorType = st.selectbox(
    "Visitor Type",
    ['Returning_Visitor', 'New_Visitor', 'Other']
)
Weekend = st.selectbox("Weekend", ["False", "True"])
 
# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
 
    # Build input as a DataFrame with correct column names
    # NOTE: these column names MUST exactly match the columns the
    # preprocessor was fit on (check preprocessor.feature_names_in_
    # if unsure, e.g. by running it once locally and printing it).
    columns = [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        "Month", "OperatingSystems", "Browser", "Region",
        "TrafficType", "VisitorType", "Weekend"
    ]
 
    row = [[
        Administrative,
        Administrative_Duration,
        Informational,
        Informational_Duration,
        ProductRelated,
        ProductRelated_Duration,
        BounceRates,
        ExitRates,
        PageValues,
        SpecialDay,
        Month,
        OperatingSystems,
        Browser,
        Region,
        TrafficType,
        VisitorType,
        Weekend == "True"
    ]]
 
    input_data = pd.DataFrame(row, columns=columns)
 
    # Debug info (safe now, since input_data is a DataFrame)
    st.write("Input columns:", input_data.columns.tolist())
    st.write("Input shape:", input_data.shape)
    if hasattr(preprocessor, "feature_names_in_"):
        st.write("Expected features:", list(preprocessor.feature_names_in_))
    if hasattr(preprocessor, "n_features_in_"):
        st.write("Expected n_features:", preprocessor.n_features_in_)
 
    # Preprocess
    processed_data = preprocessor.transform(input_data)
 
   # Feature Engineering (must match training exactly)
    VISITOR_FREQ_MAP = {
        "Returning_Visitor": 0.854650,
        "New_Visitor": 0.138714,
        "Other": 0.006637
    }
    MONTH_FREQ_MAP = {
        "May": 0.272757, "Nov": 0.244326, "Mar": 0.152397, "Dec": 0.139779,
        "Oct": 0.044982, "Sep": 0.036706, "Aug": 0.035477, "Jul": 0.035395,
        "June": 0.023351, "Feb": 0.014830
    }

    input_data["Total_Pages"] = (
        input_data["Administrative"] + input_data["Informational"] + input_data["ProductRelated"]
    )
    input_data["Total_Duration"] = (
        input_data["Administrative_Duration"] + input_data["Informational_Duration"] + input_data["ProductRelated_Duration"]
    )
    input_data["Engagement_Score"] = (
        input_data["PageValues"] / (input_data["BounceRates"] + input_data["ExitRates"] + 1e-5)
    )
    input_data["VisitorType_Freq"] = input_data["VisitorType"].map(VISITOR_FREQ_MAP)
    input_data["Month_Freq"] = input_data["Month"].map(MONTH_FREQ_MAP)

    # Preprocess
    processed_data = preprocessor.transform(input_data)
    

import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="kottanagu/smart-tourism-customer-prediction-platform", filename="best_tourism_predict_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Wellness Tourism Package Purchase Prediction Using an End-to-End MLOps Pipeline")
st.write("""
This application uses machine learning to predict the likelihood of a customer purchasing the Wellness Tourism Package.
""")

# User input
Age = st.number_input("Age", min_value=18, step=1, value=1)
Type_of_Contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
City_Tier = st.selectbox("City Tier", ["1", "2", "3"])
Duration_of_Pitch = st.number_input("Duration of Pitch", min_value=20, step=1, value=1)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Number_of_Persons_Visiting = st.number_input("Number of Persons Visiting", min_value=2, step=1, value=1)
Number_of_Followups = st.number_input("Number of Followups", min_value=2, step=1, value=1)
Product_Pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
Preferred_Property_Star = st.selectbox("Preferred Property Star", ["3", "4", "5"])
Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Number_Of_Trips = st.number_input("Number of Trips", min_value=2, step=1, value=1)
Passport = st.selectbox("Passport", ["Yes", "No"])
Pitch_Satisfaction_Score = st.selectbox("Pitch Satisfaction Score", ["1", "2", "3", "4", "5"])
Own_Car = st.selectbox("Own Car", ["Yes", "No"])
Number_Of_Children_Visiting = st.number_input("Number of Children Visiting", min_value=1, step=1, value=1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager",  "AVP", "VP"])
Monthly_Income = st.number_input("Monthly Income", min_value=25000, step=100, value=1000)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'Type_of_Contact': Type_of_Contact,
    'City_Tier': City_Tier,
    'Duration_of_Pitch': Duration_of_Pitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'Number_of_Persons_Visiting': Number_of_Persons_Visiting,
    'Number_of_Followups': Number_of_Followups,
    'Product_Pitched': Product_Pitched,
    'Preferred_Property_Star': Preferred_Property_Star,
    'Marital_Status': Marital_Status,
    'Number_Of_Trips': Number_Of_Trips,
    'Passport': Passport,
    'Pitch_Satisfaction_Score': Pitch_Satisfaction_Score,
    'Own_Car': Own_Car,
    'Number_Of_Children_Visiting': Number_Of_Children_Visiting,
    'Designation': Designation,
    'Monthly_Income': Monthly_Income
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Package Purchase Prediction" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

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
Age = st.number_input("Age", min_value=18, step=1, value=18)
Type_of_Contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
City_Tier = st.selectbox("City Tier", ["1", "2", "3"])
Duration_of_Pitch = st.number_input("Duration of Pitch", min_value=20, step=1, value=20)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Number_of_Persons_Visiting = st.number_input("Number of Persons Visiting", min_value=2, step=1, value=2)
Number_of_Followups = st.number_input("Number of Followups", min_value=2, step=1, value=2)
Product_Pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
Preferred_Property_Star = st.selectbox("Preferred Property Star", ["3", "4", "5"])
Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Number_Of_Trips = st.number_input("Number of Trips", min_value=1, step=1, value=1)
Passport = st.selectbox("Passport", ["Yes", "No"])
Pitch_Satisfaction_Score = st.selectbox("Pitch Satisfaction Score", ["1", "2", "3", "4", "5"])
Own_Car = st.selectbox("Own Car", ["Yes", "No"])
Number_Of_Children_Visiting = st.number_input("Number of Children Visiting", min_value=1, step=1, value=1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager",  "AVP", "VP"])
Monthly_Income = st.number_input("Monthly Income", min_value=10000, step=100, value=10000)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': Type_of_Contact,
    'CityTier': int(City_Tier),
    'DurationOfPitch': Duration_of_Pitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': Number_of_Persons_Visiting,
    'NumberOfFollowups': Number_of_Followups,
    'ProductPitched': Product_Pitched,
    'PreferredPropertyStar': int(Preferred_Property_Star),
    'MaritalStatus': Marital_Status,
    'NumberOfTrips': Number_Of_Trips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': int(Pitch_Satisfaction_Score),
    'OwnCar': 1 if Own_Car == "Yes" else 0,
    'NumberOfChildrenVisiting': Number_Of_Children_Visiting,
    'Designation': Designation,
    'MonthlyIncome': Monthly_Income
}])

if st.button("Predict Purchase"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        if prediction == 1:
            st.success(
                f" Great News! This customer is likely to purchase the Wellness Tourism Package"
                f"(Probability: {probability:.2f}%)."
            )
        else:
            st.error(
                f"Attention: This customer is unlikely to purchase the Wellness Tourism Package at this time. "
                f"(Probability: {probability:.2f}%)."
            )

    except Exception as e:
        st.error(f"Error: {e}")

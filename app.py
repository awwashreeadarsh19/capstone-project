import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Insurance Response Prediction",
    page_icon="🚗",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return joblib.load(
        "model/insurance_response_model.pkl"
    )


model = load_model()


# ==========================================
# TITLE
# ==========================================

st.title("🚗 Insurance Customer Response Prediction")

st.write(
    "Enter customer and vehicle information "
    "to predict the customer's response to an "
    "insurance policy offer."
)


st.divider()


# ==========================================
# CUSTOMER INFORMATION
# ==========================================

st.subheader("Customer Information")


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)


driving_license = st.selectbox(
    "Driving License",
    ["Yes", "No"]
)

driving_license = (
    1 if driving_license == "Yes" else 0
)


region_code = st.number_input(
    "Region Code",
    min_value=0.0,
    max_value=60.0,
    value=28.0,
    step=1.0
)


previously_insured = st.selectbox(
    "Previously Insured",
    ["Yes", "No"]
)

previously_insured = (
    1 if previously_insured == "Yes" else 0
)


# ==========================================
# VEHICLE INFORMATION
# ==========================================

st.subheader("Vehicle Information")


vehicle_age = st.selectbox(
    "Vehicle Age",
    [
        "< 1 Year",
        "1-2 Year",
        "> 2 Years"
    ]
)


vehicle_damage = st.selectbox(
    "Vehicle Damage",
    ["Yes", "No"]
)


annual_premium = st.number_input(
    "Annual Premium",
    min_value=0.0,
    value=35000.0,
    step=500.0
)


# ==========================================
# POLICY INFORMATION
# ==========================================

st.subheader("Policy Information")


policy_sales_channel = st.number_input(
    "Policy Sales Channel",
    min_value=1.0,
    max_value=200.0,
    value=26.0,
    step=1.0
)


vintage = st.number_input(
    "Customer Vintage (Days)",
    min_value=1,
    max_value=300,
    value=150,
    step=1
)


# ==========================================
# FEATURE ENGINEERING
# ==========================================

if vehicle_age == "< 1 Year":

    vehicle_age_years = 0.5

elif vehicle_age == "1-2 Year":

    vehicle_age_years = 1.5

else:

    vehicle_age_years = 3.0


# Age group

if age <= 25:

    age_group = "Young"

elif age <= 35:

    age_group = "Adult"

elif age <= 45:

    age_group = "Middle_Age"

elif age <= 55:

    age_group = "Senior"

else:

    age_group = "Older"


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Response",
    use_container_width=True
)


if predict_button:

    # Create input dataframe

    customer_data = pd.DataFrame({

        "Gender": [gender],

        "Age": [age],

        "Driving_License": [driving_license],

        "Region_Code": [region_code],

        "Previously_Insured": [previously_insured],

        "Vehicle_Age": [vehicle_age],

        "Vehicle_Damage": [vehicle_damage],

        "Annual_Premium": [annual_premium],

        "Policy_Sales_Channel": [
            policy_sales_channel
        ],

        "Vintage": [vintage],

        "Vehicle_Age_Years": [
            vehicle_age_years
        ],

        "Age_Group": [age_group]
    })


    # ======================================
    # PREDICTION
    # ======================================

    probability = model.predict_proba(
        customer_data
    )[0][1]


    # Classification threshold

    threshold = 0.50

    prediction = int(
        probability >= threshold
    )


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.divider()

    st.subheader("Prediction Result")


    if prediction == 1:

        st.success(
            "✅ Customer is predicted to RESPOND "
            "to the insurance offer."
        )

    else:

        st.info(
            "ℹ️ Customer is predicted NOT TO RESPOND "
            "to the insurance offer."
        )


    st.metric(
        "Response Probability",
        f"{probability * 100:.2f}%"
    )


    st.progress(
        float(probability)
    )


    st.caption(
        f"Classification threshold: {threshold:.2f}"
    )
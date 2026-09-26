import pandas as pd
import joblib
import time


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("Loading insurance prediction model...")

start = time.time()

model = joblib.load(
    "model/insurance_response_model.pkl"
)

print(
    f"Model loaded successfully in "
    f"{time.time() - start:.2f} seconds\n"
)


# ==========================================
# GET CUSTOMER INPUT
# ==========================================

print("==========================================")
print("   INSURANCE CUSTOMER RESPONSE PREDICTION")
print("==========================================\n")


gender = input(
    "Enter Gender (Male/Female): "
).strip()


age = int(
    input("Enter Age: ")
)


driving_license = int(
    input("Driving License? (1=Yes, 0=No): ")
)


region_code = float(
    input("Enter Region Code: ")
)


previously_insured = int(
    input("Previously Insured? (1=Yes, 0=No): ")
)


vehicle_age = input(
    "Enter Vehicle Age (< 1 Year / 1-2 Year / > 2 Years): "
).strip()


vehicle_damage = input(
    "Vehicle Damage? (Yes/No): "
).strip()


annual_premium = float(
    input("Enter Annual Premium: ")
)


policy_sales_channel = float(
    input("Enter Policy Sales Channel: ")
)


vintage = int(
    input("Enter Vintage: ")
)


# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Convert vehicle age into numerical value

if vehicle_age == "< 1 Year":

    vehicle_age_years = 0.5

elif vehicle_age == "1-2 Year":

    vehicle_age_years = 1.5

elif vehicle_age == "> 2 Years":

    vehicle_age_years = 3.0

else:

    print("\nInvalid Vehicle Age.")
    print("Please use:")
    print("< 1 Year")
    print("1-2 Year")
    print("> 2 Years")
    exit()


# Create age group

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
# CREATE CUSTOMER DATAFRAME
# ==========================================

customer_data = pd.DataFrame({

    "Gender": [gender],

    "Age": [age],

    "Driving_License": [driving_license],

    "Region_Code": [region_code],

    "Previously_Insured": [previously_insured],

    "Vehicle_Age": [vehicle_age],

    "Vehicle_Damage": [vehicle_damage],

    "Annual_Premium": [annual_premium],

    "Policy_Sales_Channel": [policy_sales_channel],

    "Vintage": [vintage],

    "Vehicle_Age_Years": [vehicle_age_years],

    "Age_Group": [age_group]

})


# ==========================================
# MAKE PREDICTION
# ==========================================

# Get probability only once

probability = model.predict_proba(
    customer_data
)[0][1]


# Classification threshold

threshold = 0.50

prediction = int(
    probability >= threshold
)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n==========================================")
print("              PREDICTION")
print("==========================================")

if prediction == 1:

    print("Predicted Response : YES")

else:

    print("Predicted Response : NO")


print(
    f"Response Probability: {probability * 100:.2f}%"
)

print(
    f"Classification Threshold: {threshold:.2f}"
)

print("==========================================")
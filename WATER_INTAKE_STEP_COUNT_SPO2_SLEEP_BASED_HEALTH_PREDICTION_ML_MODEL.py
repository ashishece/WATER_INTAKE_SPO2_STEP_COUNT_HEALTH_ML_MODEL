#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd Desktop


# In[2]:


cd WATER_INTAKE_STEP_COUNT_BASED_ML_MODEL_JUNE_2024_BIOARO


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# In[4]:


# Load the dataset
file_path = 'health_data.csv'
df = pd.read_csv(file_path)

# Define the diseases
diseases = ['Heart Disease', 'Diabetes', 'Hypertension']


# In[5]:


# Feature Engineering: Create nuanced disease labels based on interactions between factors
def create_disease_labels(row):
    heart_disease = (
        (row['age'] > 50) & 
        (row['weight_kg'] > 80) & 
        (row['spo2'] < 95) & 
        (row['heart_rate'] > 80)
    )
    diabetes = (
        (row['age'] > 45) & 
        (row['weight_kg'] > 90) & 
        (row['calories_intake'] > 2500)
    )
    hypertension = (
        (row['age'] > 40) & 
        (row['weight_kg'] > 85) & 
        (row['spo2'] < 95)
    )
    return pd.Series([heart_disease, diabetes, hypertension])

df[diseases] = df.apply(create_disease_labels, axis=1)

# One-hot encoding for 'gender' column
df = pd.get_dummies(df, columns=['gender'], drop_first=False)

# Splitting the dataset into features and target variables
X = df.drop(columns=['person_id'] + diseases)
y = df[diseases]

# Capture the feature names order
feature_names = X.columns.tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a MultiOutputClassifier with RandomForestClassifier
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
multi_target_model = MultiOutputClassifier(base_model, n_jobs=-1)
multi_target_model.fit(X_train, y_train)


# In[6]:


# Evaluate the model
y_pred = multi_target_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[7]:


def evaluate_user_input(sleep_hours, water_intake_glasses, spo2, step_count, weight_kg, height_cm, age, gender):
    # Define standard limits
    standard_limits = {
        'sleep_hours': 7,
        'water_intake_glasses': 8,
        'spo2': 95,
        'step_count': 10000,
    }
    
    # Define age-specific weight ranges (example ranges, adjust as needed)
    weight_ranges = {
        (0, 18): (30, 60),
        (19, 30): (50, 80),
        (31, 50): (60, 90),
        (51, 70): (55, 85),
        (71, 100): (50, 80),
    }
    
    # Determine the appropriate weight range for the given age
    weight_range = None
    for age_range, w_range in weight_ranges.items():
        if age_range[0] <= age <= age_range[1]:
            weight_range = w_range
            break

    if weight_range is None:
        weight_range = (50, 80)  # Default range if age is outside specified ranges

    # Check each parameter against the standard limit
    safe_zone = True
    if sleep_hours < standard_limits['sleep_hours']:
        print("Sleep hours are less than the standard limit.")
        safe_zone = False
    elif sleep_hours > standard_limits['sleep_hours']:
        print("Sleep hours are more than the standard limit.")
        safe_zone = False
    
    if water_intake_glasses < standard_limits['water_intake_glasses']:
        print("Water intake is less than the standard limit.")
        safe_zone = False
    elif water_intake_glasses > standard_limits['water_intake_glasses']:
        print("Water intake is more than the standard limit.")
        safe_zone = False

    if spo2 < standard_limits['spo2']:
        print("SPO2 level is less than the standard limit.")
        safe_zone = False
    
    if step_count < standard_limits['step_count']:
        print("Step count is less than the standard limit.")
        safe_zone = False
    elif step_count > standard_limits['step_count']:
        print("Step count is more than the standard limit.")
        safe_zone = False
    
    if weight_kg < weight_range[0]:
        print(f"Weight is less than the standard range ({weight_range[0]} - {weight_range[1]} kg) for your age.")
        safe_zone = False
    elif weight_kg > weight_range[1]:
        print(f"Weight is more than the standard range ({weight_range[0]} - {weight_range[1]} kg) for your age.")
        safe_zone = False
    
    if safe_zone:
        print("User is in the safe zone.")
    else:
        print("User is in the danger zone.")
    
    return safe_zone


# In[8]:


def predict_health_risk(model, feature_names, sleep_hours, water_intake_glasses, spo2, step_count, weight_kg, height_cm, age, gender):
    # One-hot encode gender manually to ensure it matches training data
    gender_dict = {'gender_Male': 0, 'gender_Female': 0, 'gender_Other': 0}
    gender_key = f'gender_{gender}'
    if gender_key in gender_dict:
        gender_dict[gender_key] = 1
    
    user_data = pd.DataFrame({
        'age': [age],
        'sleep_hours': [sleep_hours],
        'water_intake_glasses': [water_intake_glasses],
        'spo2': [spo2],
        'step_count': [step_count],
        'weight_kg': [weight_kg],
        'height_cm': [height_cm],
        'calories_intake': [2500],  # Assume average value for simplicity
        'exercise_minutes': [30],  # Assume average value for simplicity
        'heart_rate': [70],  # Assume average value for simplicity
        'gender_Male': [gender_dict['gender_Male']],
        'gender_Female': [gender_dict['gender_Female']],
        'gender_Other': [gender_dict['gender_Other']]
    })
    
    # Reorder the user_data to match the feature_names
    user_data = user_data[feature_names]
    
    risk_probabilities = model.predict_proba(user_data)
    risk_probabilities = [prob[0][1] for prob in risk_probabilities]  # Extract the probability for the positive class
    
    disease_risk = {disease: risk for disease, risk in zip(diseases, risk_probabilities)}

    for disease, risk in disease_risk.items():
        print(f"Risk of {disease}: {risk * 100:.2f}%")
    
    return disease_risk


# In[9]:


def provide_recommendations(disease_risk, sleep_hours, water_intake_glasses, step_count, weight_kg, age):
    standard_limits = {
        'sleep_hours': 7,
        'water_intake_glasses': 8,
        'step_count': 10000,
    }
    
    # Define age-specific weight ranges (example ranges, adjust as needed)
    weight_ranges = {
        (0, 18): (30, 60),
        (19, 30): (50, 80),
        (31, 50): (60, 90),
        (51, 70): (55, 85),
        (71, 100): (50, 80),
    }
    
    # Determine the appropriate weight range for the given age
    weight_range = None
    for age_range, w_range in weight_ranges.items():
        if age_range[0] <= age <= age_range[1]:
            weight_range = w_range
            break

    if weight_range is None:
        weight_range = (50, 80)  # Default range if age is outside specified ranges

    disease_recommendations = {
        'Heart Disease': [
            "Increase physical activity.",
            "Eat a heart-healthy diet.",
            "Maintain a healthy weight.",
            "Avoid smoking and excessive alcohol intake.",
            "Monitor blood pressure and cholesterol levels."
        ],
        'Diabetes': [
            "Monitor blood sugar levels regularly.",
            "Follow a balanced diet rich in fiber and low in sugars.",
            "Exercise regularly to maintain a healthy weight.",
            "Stay hydrated by drinking plenty of water.",
            "Get regular medical check-ups."
        ],
        'Hypertension': [
            "Reduce salt intake in your diet.",
            "Maintain a healthy weight.",
            "Exercise regularly.",
            "Limit alcohol consumption.",
            "Manage stress through relaxation techniques."
        ]
    }
    
    # Print disease risk and recommendations
    for disease, risk in disease_risk.items():
        print(f"\nRisk of {disease}: {risk * 100:.2f}%")
        if risk > 0.0:  # Provide recommendations for any non-zero risk
            print("Recommendations to reduce risk:")
            for recommendation in disease_recommendations[disease]:
                print(f"- {recommendation}")

    # Separate recommendations for specific health parameters
    print("\nAdditional Recommendations:")
    
    if sleep_hours < standard_limits['sleep_hours']:
        print(f"- Sleep at least {standard_limits['sleep_hours']} hours per night to meet the standard.")
    elif sleep_hours > standard_limits['sleep_hours']:
        print(f"- Reduce sleep to around {standard_limits['sleep_hours']} hours per night to meet the standard.")

    if water_intake_glasses < standard_limits['water_intake_glasses']:
        additional_glasses = standard_limits['water_intake_glasses'] - water_intake_glasses
        print(f"- Gradually increase your water intake by 1-2 glasses, aiming for {standard_limits['water_intake_glasses']} glasses per day. Current target: {water_intake_glasses + 1} glasses.")
    elif water_intake_glasses > standard_limits['water_intake_glasses']:
        print(f"- Reduce your water intake to meet the standard of {standard_limits['water_intake_glasses']} glasses per day.")
    
    if step_count < standard_limits['step_count']:
        print(f"- Gradually increase your step count by 1000 steps each week, aiming for {standard_limits['step_count']} steps per day. Current target: {step_count + 1000} steps.")
    elif step_count > standard_limits['step_count']:
        print(f"- Reduce your step count to meet the standard of {standard_limits['step_count']} steps per day.")
    
    if weight_kg < weight_range[0]:
        print(f"- Increase your weight to be within the standard range ({weight_range[0]} - {weight_range[1]} kg) for your age.")
        print("- Consume calorie-dense nutritious foods like nuts, seeds, and dairy products.")
        print("- Include strength training exercises in your routine.")
    elif weight_kg > weight_range[1]:
        print(f"- Reduce your weight to be within the standard range ({weight_range[0]} - {weight_range[1]} kg) for your age.")
        print("- Follow a balanced diet with controlled portions.")
        print("- Increase your physical activity, focusing on both cardio and strength training.")
        print("- Avoid high-calorie, low-nutrient foods.")


# In[ ]:


# Get user input
sleep_hours = float(input("Enter your average sleep hours per night: "))
water_intake_glasses = float(input("Enter your average daily water intake (in glasses): "))
spo2 = float(input("Enter your SPO2 level: "))
step_count = int(input("Enter your average daily step count: "))
weight_kg = float(input("Enter your weight (in kg): "))
height_cm = float(input("Enter your height (in cm): "))
age = int(input("Enter your age: "))
gender = input("Enter your gender (Male/Female/Other): ")

# Evaluate user input
safe_zone = evaluate_user_input(sleep_hours, water_intake_glasses, spo2, step_count, weight_kg, height_cm, age, gender)

# Predict health risk
disease_risk = predict_health_risk(multi_target_model, feature_names, sleep_hours, water_intake_glasses, spo2, step_count, weight_kg, height_cm, age, gender)

# Provide recommendations
provide_recommendations(disease_risk, sleep_hours, water_intake_glasses, step_count, weight_kg, age)


# In[ ]:





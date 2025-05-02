import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# หัวข้อแอป
st.title("App predict the likelihoof that a student will drop out")

# คำอธิบาย
st.write("input data below")

# โหลดข้อมูล
data = pd.read_csv("Deploy File - ชีต1 (2).csv")

# เลือกตัวแปร
features = ['Monthly education expenses', 'Absences per month', 'Engagement Score',
            'economie_factor', 'motivational_factors', 'family_factors',
            'social_factors', 'distance_factor', 'health_factor', 'school_environment_factor']
X = data[features]
y = data['Final decision to drop out']

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale ข้อมูล
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างและฝึกโมเดล
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ฟังก์ชันแปลงตัวเลขเป็นข้อความ
def convert_to_text(value, factor_type):
    if factor_type == 'expense':
        if value == 0:
            return "1 million to 2 million kip"
        elif value == 1:
            return "100,000 to 500,000 kip"
        elif value == 2:
            return "500,000 - 1 million kip"
        elif value == 3:
            return "More than 2 million kip"
        elif value == 4:
            return "less than 100,000 kip"
    elif factor_type in ['factor', 'absence']:
        if factor_type == 'absence':
            return f"{value} day"
        elif value == 1:
            return "Not much"
        elif value == 2:
            return "Rarely"
        elif value == 3:
            return "Moderate"
        elif value == 4:
            return "Much"
        elif value == 5:
            return "Very much"
    elif factor_type == 'engagement':
        return str(value)
    return str(value)

# อินพุตจากผู้ใช้
st.subheader("input data student")

# Education Expenses
expense_options = {
    "1 to 2 million kip": 0,
    "100,000 to 500,000 kip": 1,
    "500,000 to 1 million kip": 2,
    "More than 2 million kip": 3,
    "Less than 100,000 kip": 4
}
monthly_expenses_text = st.selectbox("Education Expenses", list(expense_options.keys()))
monthly_expenses = expense_options[monthly_expenses_text]

# Number of school absence days
absences = st.number_input("Number of school absence days per month", min_value=0, value=2, step=1)

# Learning engagement score
engagement_score = st.slider("Learning Engagement Score (-4 to 8)", -4, 8, 0)

# Contributing factors
factor_options = ["Not much", "Rarely", "Moderate", "Much", "Very much"]
factor_values = {text: i+1 for i, text in enumerate(factor_options)}

economie_factor_text = st.selectbox("Economic Factors", factor_options)
economie_factor = factor_values[economie_factor_text]

motivational_factors_text = st.selectbox("Motivational Factors", factor_options)
motivational_factors = factor_values[motivational_factors_text]

family_factors_text = st.selectbox("Family Factors", factor_options)
family_factors = factor_values[family_factors_text]

social_factors_text = st.selectbox("Social Factors", factor_options)
social_factors = factor_values[social_factors_text]

distance_factor_text = st.selectbox("Distance Factors", factor_options)
distance_factor = factor_values[distance_factor_text]

health_factor_text = st.selectbox("Health Factors", factor_options)
health_factor = factor_values[health_factor_text]

school_environment_factor_text = st.selectbox("School Environment Factors", factor_options)
school_environment_factor = factor_values[school_environment_factor_text]

# Combine user input
user_input = np.array([[monthly_expenses, absences, engagement_score, 
                        economie_factor, motivational_factors, family_factors,
                        social_factors, distance_factor, health_factor, school_environment_factor]])

# Scale input
user_input_scaled = scaler.transform(user_input)

# Predict
prediction = model.predict(user_input_scaled)
prediction_prob = model.predict_proba(user_input_scaled)[0][1] * 100  # Dropout probability (class 1)

# Prediction result
st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error(f"⚠️ High risk of dropping out ({prediction_prob:.1f}% chance of dropout)")
else:
    st.success(f"✅ Low risk of dropping out ({prediction_prob:.1f}% chance of dropout)")

# Display user input
st.subheader("Input Summary")
st.write(f"- Education Expenses: {monthly_expenses_text}")
st.write(f"- Absence Days per Month: {convert_to_text(absences, 'absence')}")
st.write(f"- Engagement Score: {convert_to_text(engagement_score, 'engagement')}")
st.write(f"- Economic Factors: {economie_factor_text}")
st.write(f"- Motivational Factors: {motivational_factors_text}")
st.write(f"- Family Factors: {family_factors_text}")
st.write(f"- Social Factors: {social_factors_text}")
st.write(f"- Distance Factors: {distance_factor_text}")
st.write(f"- Health Factors: {health_factor_text}")
st.write(f"- School Environment Factors: {school_environment_factor_text}")

# Suggestions
st.subheader("Recommendation")
if prediction_prob > 70:
    st.write("The student should be closely monitored and provided with academic or family counseling.")
elif prediction_prob > 50:
    st.write("The student should be given additional guidance and reviewed for possible risk factors.")
else:
    st.write("The student shows strong potential to continue their education. Support and encourage their motivation.")

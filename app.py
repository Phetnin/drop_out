import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# หัวข้อแอป
st.title("แอปทำนายโอกาสดรอปเรียนสำหรับนักเรียน")

# คำอธิบาย
st.write("กรอกข้อมูลด้านล่างเพื่อทำนายโอกาสที่นักเรียนจะดรอปเรียน")

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

# อินพุตจากผู้ใช้
st.subheader("กรอกข้อมูลนักเรียน")
monthly_expenses = st.slider("ค่าใช้จ่ายในการศึกษา (0-4)", 0, 4, 2)  # ปรับช่วงเป็น 0-4
absences = st.number_input("จำนวนวันที่ขาดเรียนต่อเดือน", min_value=0, value=2, step=1)
engagement_score = st.slider("คะแนนความสนใจในการเรียน (-2 ถึง 3)", -2, 3, 0)  # ปรับช่วงเป็น -2 ถึง 3
economie_factor = st.slider("ปัจจัยด้านเศรษฐกิจ (0-4)", 0, 4, 2)  # ปรับช่วงเป็น 0-4
motivational_factors = st.slider("ปัจจัยด้านแรงจูงใจ (0-4)", 0, 4, 2)  # ปรับช่วงเป็น 0-4
family_factors = st.slider("ปัจจัยด้านครอบครัว (0-4)", 0, 4, 2)  # ปรับช่วงเป็น 0-4
social_factors = st.slider("ปัจจัยด้านสังคม (0-4)", 0, 4, 2)  # ปรับช่วงเป็น 0-4
distance_factor = st.slider("ปัจจัยด้านระยะทาง (0-4)", 0, 4, 2)  # ปรับช่วงเป็น 0-4
health_factor = st.slider("ปัจจัยด้านสุขภาพ (0-4)", 0, 4, 2)  # ปรับช่วงเป็น 0-4
school_environment_factor = st.slider("ปัจจัยด้านสภาพแวดล้อมโรงเรียน (0-4)", 0, 4, 2)  # ปรับช่วงเป็น 0-4

# รวมข้อมูลที่ผู้ใช้กรอก
user_input = np.array([[monthly_expenses, absences, engagement_score, 
                        economie_factor, motivational_factors, family_factors,
                        social_factors, distance_factor, health_factor, school_environment_factor]])

# Scale ข้อมูลผู้ใช้
user_input_scaled = scaler.transform(user_input)

# ทำนาย
prediction = model.predict(user_input_scaled)
prediction_prob = model.predict_proba(user_input_scaled)[0][1] * 100  # ความน่าจะเป็นของการดรอป (class 1)

# แสดงผลลัพธ์
st.subheader("ผลการทำนาย")
if prediction[0] == 1:
    st.error(f"⚠️ นักเรียนมีโอกาสดรอปเรียนสูง ({prediction_prob:.1f}% โอกาสดรอปเรียน)")
else:
    st.success(f"✅ นักเรียนมีโอกาสดรอปเรียนต่ำ ({prediction_prob:.1f}% โอกาสดรอปเรียน)")

# คำแนะนำ
st.subheader("คำแนะนำ")
if prediction_prob > 70:
    st.write("ควรติดตามนักเรียนอย่างใกล้ชิด และให้คำปรึกษาด้านการเรียนหรือครอบครัว")
elif prediction_prob > 50:
    st.write("ควรให้คำแนะนำเพิ่มเติม และตรวจสอบปัจจัยที่อาจส่งผลต่อการเรียน")
else:
    st.write("นักเรียนมีแนวโน้มที่ดีในการเรียนต่อ ควรสนับสนุนให้รักษาความตั้งใจ")

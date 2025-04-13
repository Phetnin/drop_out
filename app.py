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
data = pd.read_csv("Full data set 515 - หลัก1.csv")

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
            return "1 ล้านถึง 2 ล้านบาท"
        elif value == 1:
            return "100,000 ถึง 500,000 บาท"
        elif value == 2:
            return "500,000 ถึง 1 ล้านบาท"
        elif value == 3:
            return "มากกว่า 2 ล้านบาท"
        elif value == 4:
            return "น้อยกว่า 100,000 บาท"
    elif factor_type in ['factor', 'absence']:
        if factor_type == 'absence':
            return f"{value} วัน"
        elif value == 1:
            return "ไม่มาก"
        elif value == 2:
            return "ไม่ค่อย"
        elif value == 3:
            return "ปานกลาง"
        elif value == 4:
            return "มาก"
        elif value == 5:
            return "เยอะมาก"
    elif factor_type == 'engagement':
        return str(value)
    return str(value)

# อินพุตจากผู้ใช้
st.subheader("กรอกข้อมูลนักเรียน")

# ค่าใช้จ่ายในการศึกษา
expense_options = {
    "1 ล้านถึง 2 ล้านบาท": 0,
    "100,000 ถึง 500,000 บาท": 1,
    "500,000 ถึง 1 ล้านบาท": 2,
    "มากกว่า 2 ล้านบาท": 3,
    "น้อยกว่า 100,000 บาท": 4
}
monthly_expenses_text = st.selectbox("ค่าใช้จ่ายในการศึกษา", list(expense_options.keys()))
monthly_expenses = expense_options[monthly_expenses_text]

# จำนวนวันที่ขาดเรียน
absences = st.number_input("จำนวนวันที่ขาดเรียนต่อเดือน", min_value=0, value=2, step=1)

# คะแนนความสนใจในการเรียน
engagement_score = st.slider("คะแนนความสนใจในการเรียน (-4 ถึง 8)", -4, 8, 0)

# ปัจจัยต่าง ๆ
factor_options = ["ไม่มาก", "ไม่ค่อย", "ปานกลาง", "มาก", "เยอะมาก"]
factor_values = {text: i+1 for i, text in enumerate(factor_options)}

economie_factor_text = st.selectbox("ปัจจัยด้านเศรษฐกิจ", factor_options)
economie_factor = factor_values[economie_factor_text]

motivational_factors_text = st.selectbox("ปัจจัยด้านแรงจูงใจ", factor_options)
motivational_factors = factor_values[motivational_factors_text]

family_factors_text = st.selectbox("ปัจจัยด้านครอบครัว", factor_options)
family_factors = factor_values[family_factors_text]

social_factors_text = st.selectbox("ปัจจัยด้านสังคม", factor_options)
social_factors = factor_values[social_factors_text]

distance_factor_text = st.selectbox("ปัจจัยด้านระยะทาง", factor_options)
distance_factor = factor_values[distance_factor_text]

health_factor_text = st.selectbox("ปัจจัยด้านสุขภาพ", factor_options)
health_factor = factor_values[health_factor_text]

school_environment_factor_text = st.selectbox("ปัจจัยด้านสภาพแวดล้อมโรงเรียน", factor_options)
school_environment_factor = factor_values[school_environment_factor_text]

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

# แสดงข้อมูลที่ผู้ใช้กรอก
st.subheader("ข้อมูลที่กรอก")
st.write(f"- ค่าใช้จ่ายในการศึกษา: {monthly_expenses_text}")
st.write(f"- จำนวนวันที่ขาดเรียนต่อเดือน: {convert_to_text(absences, 'absence')}")
st.write(f"- คะแนนความสนใจในการเรียน: {convert_to_text(engagement_score, 'engagement')}")
st.write(f"- ปัจจัยด้านเศรษฐกิจ: {economie_factor_text}")
st.write(f"- ปัจจัยด้านแรงจูงใจ: {motivational_factors_text}")
st.write(f"- ปัจจัยด้านครอบครัว: {family_factors_text}")
st.write(f"- ปัจจัยด้านสังคม: {social_factors_text}")
st.write(f"- ปัจจัยด้านระยะทาง: {distance_factor_text}")
st.write(f"- ปัจจัยด้านสุขภาพ: {health_factor_text}")
st.write(f"- ปัจจัยด้านสภาพแวดล้อมโรงเรียน: {school_environment_factor_text}")

# คำแนะนำ
st.subheader("คำแนะนำ")
if prediction_prob > 70:
    st.write("ควรติดตามนักเรียนอย่างใกล้ชิด และให้คำปรึกษาด้านการเรียนหรือครอบครัว")
elif prediction_prob > 50:
    st.write("ควรให้คำแนะนำเพิ่มเติม และตรวจสอบปัจจัยที่อาจส่งผลต่อการเรียน")
else:
    st.write("นักเรียนมีแนวโน้มที่ดีในการเรียนต่อ ควรสนับสนุนให้รักษาความตั้งใจ")

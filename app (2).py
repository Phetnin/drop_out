import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
import plotly.express as px

# หัวข้อแอป
st.title("แอปทำนายการดรอปเรียน")

# โหลดข้อมูล
data = pd.read_csv("Full data set 515 - ชีต1.csv")  # เปลี่ยนชื่อไฟล์ถ้าชื่อต่าง

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

# ทำนาย
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# แสดงผลลัพธ์
st.subheader("ผลลัพธ์โมเดล")
st.write("รายงานผล Random Forest:")
st.text(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
st.write(f"ROC-AUC Score: {roc_auc:.2f}")

# วาด ROC Curve ด้วย Plotly
fig_roc = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})',
                  labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess')
st.plotly_chart(fig_roc)

# Feature Importance
st.subheader("ความสำคัญของตัวแปร")
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig_fi = px.bar(x=feature_importance[sorted_idx], y=[features[i] for i in sorted_idx],
                orientation='h', title='Feature Importance',
                labels={'x': 'Importance', 'y': 'Feature'})
st.plotly_chart(fig_fi)

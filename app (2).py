import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
import plotly.express as px

df = pd.read_csv('Full data set 515 - ชีต1.csv')  # เปลี่ยนชื่อไฟล์ให้ตรงกับที่คุณอัปโหลด
df.duplicated().sum()
# ลบข้อมูลที่ซ้ำ
df_cleaned = df.drop_duplicates()
print("ข้อมูลหลังลบค่าซ้ำ:\n", df_cleaned)
import numpy as np

# คำนวณ Q1, Q3 และ IQR
Q1 = df["ໃນເດືອນ 1 ເຈົ້າຂາດຮຽນປະມານຈັກມື້? ໝາຍເຫດ: ໃຫ້ພິມສະເພາະເປັນໂຕເລກ ເຊັ່ນ: 0"].quantile(0.25)
Q3 = df["ໃນເດືອນ 1 ເຈົ້າຂາດຮຽນປະມານຈັກມື້? ໝາຍເຫດ: ໃຫ້ພິມສະເພາະເປັນໂຕເລກ ເຊັ່ນ: 0"].quantile(0.75)
IQR = Q3 - Q1

# กำหนดขอบเขตล่างและบน
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# แทนค่า Outliers ด้วยขอบเขต
df["ໃນເດືອນ 1 ເຈົ້າຂາດຮຽນປະມານຈັກມື້? ໝາຍເຫດ: ໃຫ້ພິມສະເພາະເປັນໂຕເລກ ເຊັ່ນ: 0"] = np.clip(df["ໃນເດືອນ 1 ເຈົ້າຂາດຮຽນປະມານຈັກມື້? ໝາຍເຫດ: ໃຫ້ພິມສະເພາະເປັນໂຕເລກ ເຊັ່ນ: 0"], lower_bound, upper_bound)
# แสดงข้อมูลหลังจาก Capping
print("Data after Winsorization:")
print(df['ໃນເດືອນ 1 ເຈົ້າຂາດຮຽນປະມານຈັກມື້? ໝາຍເຫດ: ໃຫ້ພິມສະເພາະເປັນໂຕເລກ ເຊັ່ນ: 0'])
df = df.rename(columns={'ເພດ': 'Gender'})
df = df.rename(columns={'ອາຍຸ': 'Age'})
df = df.rename(columns={'ລະດັບການສຶກສາ': 'Education level'})
df = df.rename(columns={'ໃນເດືອນ 1 ເຈົ້າຂາດຮຽນປະມານຈັກມື້? ໝາຍເຫດ: ໃຫ້ພິມສະເພາະເປັນໂຕເລກ ເຊັ່ນ: 0': 'Absences per month'})
df = df.rename(columns={'ເຈົ້າພໍໃຈກັບເນື້ອຫາຫຼັກສູດທີ່ເຈົ້າຮຽນໃນປັດຈຸບັນຫຼືບໍ່?': 'Satisfied with the curriculum'})
df = df.rename(columns={'ເຈົ້າຄິດວ່າເນື້ອຫາໃນຫຼັກສູດຕົງກັບຄວາມສົນໃຈຂອງເຈົ້າຫຼຶບໍ່?': 'Course content matching interests','ເຈົ້າຄິດວ່າເນື້ອຫາໃນຫຼັກສູດການຮຽນຍາກເກີນໄປຫຼືບໍ່?':'Course content difficulty'})
df = df.rename(columns={'ລາຍໄດ້ຂອງຄອບຄົວປະມານເທົ່າໃດ?': 'Family income',
                        'ລາຍໄດ້ຫຼັກຂອງຄອບຄົວທ່ານມາຈາກອາຊີບໃດ?':'Main family occupation',
                        'ໃນຄອບຄົວເຈົ້າໃຊ້ຈ່າຍໃນເລື່ອງໃດຫຼາຍທີ່ສຸດ?':'Familys main expenses',
                        'ຄອບຄົວຂອງທ່ານໃຊ້ຈ່າຍກັບການຮຽນຂອງທ່ານເທົ່າໃດຕໍ່ເດືອນ?':'Monthly education expenses',
                        'ເຈົ້າມີອ້າຍເອື້ຍນ້ອງຈັກຄົນ? ໝາຍເຫດ: ບໍ່ນັບຕົນນເອງ':'Number of siblings',
                        'ຖ້າຄອບຄົວຂອງເຈົ້າມີລາຍໄດ້ບໍ່ພຽງພໍຕໍ່ການລ້ຽງຊີບ':'Q1',
                        'ຖ້າຜູ້ປົກຄອງຂອງເຈົ້າບັງຄັບໃຫ້ເຮັດວຽກແທນການຮຽນ':'Q2',
                        'ຖ້າເຈົ້າມີໂອກາດໄດ້ເຮັດວຽກທີ່ໄດ້ເງິນເດີອນໂດຍບໍ່ຕ້ອງໄດ້ຮຽນ':'Q3',
                        'ຖ້າຄອບຄົວຂອງເຈົ້າມີໜີ້ສິນຫຼາຍ':'Q4',
                        'ຖ້າເຈົ້າຮູ້ສືກວ່າພາລະຄ່າໃຊ້ຈ່າຍໃນການສຶກສາເຮັດໃຫ້ຄອບຄົວມີຄວາມຄຽດ ຫຼື ຄວາມຍາກລຳບາກໃນການຈັດການການເງິນ':'Q5',
                        'ຖ້າໂຮງຮຽນບໍ່ມີຄູສອນພຽງພໍ ຫຼື ການສອນບໍ່ມີຄຸນນະພາບ':'Q6',
                        'ຖ້າໂຮງຮຽນເນັ້ນແຕ່ການສອບເສັງ ແລະ ບໍ່ມີການຮຽນຮູ້ແບບມ່ວນຊື່ນ':'Q7',
                        'ຖ້າທາງໂຮງຮຽນໄດ້ມີນະໂຍບາຍການຫາງານໃຫ້ກັບນັກຮຽນ':'Q8',
                        'ຖ້າຫຼັກສູດບໍ່ມີວິຊາທີ່ຊ່ວຍໃຫ້ກຽມຕົວສຳລັບວຽກໃນອະນາຄົດ':'Q9',
                        'ຖ້າໂຮງຮຽນບໍ່ມີສິ່ງອຳນວຍຄວາມສະດວກທີ່ຈຳເປັນເຊັ່ນ: ຫ້ອງທົດລອງ, ຫ້ອງສະໝຸດ ຫຼື ອຸປະກອນກິລາ…':'Q10',
                        'ຖ້ານັກຮຽນຮູ້ສຶກວ່າການຮຽນໃນໂຮງຮຽນບໍ່ຊ່ວຍໃຫ້ມີວຽກເຮັດໃນອະນາຄົດ':'Q11',
                        'ນັກຮຽນຮູ້ສຶກເບື່ອໜ່າຍກັບການຮຽນໃນໂຮງຮຽນ':'Q12',
                        'ຖ້ານັກຮຽນຕ້ອງຮຽນວິຊາທີ່ຕົວເອງບໍ່ມັກເປັນຈຳນວນຫຼາຍ':'Q13',
                        'ຖ້ານັກຮຽນບໍ່ມີເປົ້າໝາຍ ຫຼື ແຮງບັນດານໃຈໃນການຮຽນ':'Q14',
                        'ຖ້ານັກຮຽນຮູ້ສຶກວ່າຫຼັກສູດບໍ່ເປີດໂອກາດໃຫ້ເລືອກວິຊາທີ່ຕົວເອງ':'Q15',
                        'ຖ້ານັກຮຽນຕ້ອງຢ່າງໄປໂຮງຮຽນໄກກວ່າ 3 ກິໂລແມັດທຸກມື້':'Q16',
                        'ຖ້າຖະໜົນທີ່ເດີນທາງໄປໂຮງຮຽນບໍ່ມີຄວາມປອດໄພເຊັ່ນ: ມີອຸບັດຕິເຫດ ຫຼື ອາດຊະຍາກຳ':'Q17',
                        'ຖ້ານັກຮຽນເດີນທາງໄປໂຮງຮຽນມີຄ່າໃຊ້ຈ່າຍສູງຈົນເຮັດໃຫ້ຄອບຄົວລຳບາກທາງການເງິນ':'Q18',
                        'ຖ້າການຂົນສົ່ງໄປໂຮງຮຽນບໍ່ສະດວກ ຫຼື ມີຄວາມສ່ຽງ':'Q19',
                        'ຖ້ານັກຮຽນບໍ່ໄດ້ຮັບການສະໜັບສະໜູນຄ່າໃຊ້ຈ່າຍໃນການເດີນທາງຈາກຄອບຄົວຫຼືໂຮງຮຽນ':'Q20',
                        'ຖ້ານັກຮຽນມີພະຍາດຊຳເຮື້ອທີ່ເຮັດໃຫ້ຕ້ອງມີການຂາດຮຽນຢູ່ເລື້ອຍໆ':'Q21',
                        'ຖ້ານັກຮຽນມີບັນຫາດ້ານສຸຂະພາບຈິດເຊັ່ນ: ຄວາມຄຽດ ຫຼື ພະຍາດຊຶມເສົ້າ':'Q22',
                        'ຖ້ານັກຮຽນບໍ່ມີສິດເຂົ້າເຖິງບໍລິການສາທາລະນະສຸກທີ່ດີເຊັ່ນ: ບໍ່ມີເງິນຈ່າຍຄ່າປິ່ນປົວ':'Q23',
                        'ຖ້ານັກຮຽນມີພາລະຕ້ອງດູແລສຸຂະພາບຂອງສະມາຊິກໃນຄອບຄົວເຊັ່ນ: ພໍ່ແມ່ ຫຼື ພີ່ນ້ອງທີ່ປ່ວຍ':'Q24',
                        'ຖ້ານັກຮຽນມີບັນຫາການນອນຫຼັບທີ່ສົ່ງຜົກກະທົບຕໍ່ການຮຽນຮູ້ ແລະ ຄວາມສາມາດໃນການສຸມໃສ່':'Q25',
                        'ນັກຮຽນເຄີຍຄິດທີ່ຈະປະລະການຮຽນຍ້ອນເລື່ອງຄວາມຮັກ':'Q26',
                        'ຖ້ານັກຮຽນບໍ່ມີໝູ່ ຫຼື ຖືກກີດກັນຈາກສັງຄົມໃນໂຮງຮຽນ':'Q27',
                        'ຖ້ານັກຮຽນຂາດແຮງຈູງໃຈໃນການຮຽນ ເພະບໍ່ຮູ້ສຶກວ່າສິ່ງທີ່ຮຽນຈະເປັນປະໂຫຍດຕໍ່ຊີວິດ':'Q28',
                        'ຖ້າວ່ານັກຮຽນຮູ້ສຶກວ່າຕົນເອງບໍ່ເກັ່ງເທົ່າໝູ່ ຈົນເຮັດໃຫ້ໝົດກຳລັງໃຈໃນການຮຽນ':'Q29',
                        'ຖ້ານັກຮຽນຮູ້ສຶກວ່າຄູບໍ່ສະໜັບສະໜູນ ຫຼືບໍ່ເຂົ້າໃຈບັນຫາຂອງຕົນເອງ':'Q30',
                        'ຖ້າຄອບຄົວຂອງນັກຮຽນມີບັນຫາດ້ານຄອບຄົວ (ເຊັ່ນ:​ ການຢ່າຮ້າງ, ຄວາມຂັດແຍ່ງພາຍໃນ)':'Q31',
                        'ຖ້າຜູ້ປົກຄອງຂອງນັກຮຽນມີທັດສະນະຄະຕິທີ່ບໍ່ສະໜັບສະໜູນການສຶກສາ':'Q32',
                        'ຖ້າຄອບຄົວຂອງນັກຮຽນມີບັນຫາສຸຂະພາບ (ເຊັ່ນ: ເປັນພະຍາດຊ້ຳເຮື້ອພາຍໃນສະມາຊິກຄອບຄົວ)':'Q33',
                        'ຖ້າຄອບຄົວຂອງນັກຮຽນມີຄວາມບໍ່ສະຖຽນທາງດ້ານການເງິນ':'Q34',
                        'ຖ້ານັກຮຽນບໍ່ໄດ້ຮັບການສະໜັບສະໜູນຈາກຄອບຄົວໃນເລື່ອງການສຶກສາ ແລະ ພັດທະນາອາຊີບໃນອະນາຄົດ':'Q35',
                        'ຖ້າເຈົ້າເຈິສະຖານະການທີ່ກ່າງມາທັງໝົດນັ້ນ ເຈົ້າຈະຕັດສິນໃຈພິຈາລະນາ ປະລະການຮຽນຫຼືບໍ່?':'Final decision to drop out'})
# ดูข้อมูลเริ่มต้น
print("ข้อมูลก่อนแก้ไข:")
df['Satisfied with the curriculum'].head(10)

# ฟังก์ชันสำหรับแปลงข้อมูล
def clean_expenses(value):
    if isinstance(value, str):  # ถ้าเป็นข้อความ
        if 'ປານກາງ' in value:
            return 2  # กำหนดให้เป็น 2,000,000
        elif 'ບໍ່ພໍໃຈເລີຍ' in value:
            return 0  # ค่าเฉลี่ยของช่วง
        elif 'ໜ້ອຍ' in value:
            return 1  # ค่าเฉลี่ยของช่วง
        elif 'ພໍໃຈ' in value:
            return 3  # ค่าเฉลี่ยของช่วง
        elif 'ພໍໃຈຫຼາຍທີ່ສຸດ' in value:
            return 4  # ค่าเฉลี่ยของช่วง
        else:
            try:
                return int(value)  # แปลงเป็นตัวเลขถ้าเป็นไปได้
            except ValueError:
                return value  # คงค่าเดิมถ้าแปลงไม่ได้
    return value  # คงค่าเดิมถ้าไม่ใช่ข้อความ

# ใช้ฟังก์ชันกับคอลัมน์
df['Satisfied with the curriculum'] = df['Satisfied with the curriculum'].apply(clean_expenses)

# ดูข้อมูลหลังแก้ไข
print("\nข้อมูลหลังแก้ไข:")
print(df['Satisfied with the curriculum'].head(10))

# บันทึกกลับไปที่ไฟล์ Excel (ถ้าต้องการ)
df.to_csv('Full data set 515 - ชีต1.csv', index=False)
# ดูข้อมูลเริ่มต้น
print("ข้อมูลก่อนแก้ไข:")
df['Course content matching interests'].head(10)

# ฟังก์ชันสำหรับแปลงข้อมูล
def clean_expenses(value):
    if isinstance(value, str):  # ถ้าเป็นข้อความ
        if 'ປານກາງ' in value:
            return 2  # กำหนดให้เป็น 2,000,000
        elif 'ບໍ່ຕົງເລີຍ' in value:
            return 0  # ค่าเฉลี่ยของช่วง
        elif 'ບໍ່ຕົງ' in value:
            return 1  # ค่าเฉลี่ยของช่วง
        elif 'ຕົງ' in value:
            return 3  # ค่าเฉลี่ยของช่วง
        elif 'ຕົງທີ່ສຸດ' in value:
            return 4  # ค่าเฉลี่ยของช่วง
        else:
            try:
                return int(value)  # แปลงเป็นตัวเลขถ้าเป็นไปได้
            except ValueError:
                return value  # คงค่าเดิมถ้าแปลงไม่ได้
    return value  # คงค่าเดิมถ้าไม่ใช่ข้อความ

# ใช้ฟังก์ชันกับคอลัมน์
df['Course content matching interests'] = df['Course content matching interests'].apply(clean_expenses)

# ดูข้อมูลหลังแก้ไข
print("\nข้อมูลหลังแก้ไข:")
print(df['Course content matching interests'].head(10))

# บันทึกกลับไปที่ไฟล์ Excel (ถ้าต้องการ)
df.to_csv('Full data set 515 - ชีต1.csv', index=False)
# ดูข้อมูลเริ่มต้น
print("ข้อมูลก่อนแก้ไข:")
df['Course content difficulty'].head(10)

# ฟังก์ชันสำหรับแปลงข้อมูล
def clean_expenses(value):
    if isinstance(value, str):  # ถ้าเป็นข้อความ
        if 'ປານກາງ' in value:
            return 2  # กำหนดให้เป็น 2,000,000
        elif 'ຍາກເກີນໄປ' in value:
            return 0  # ค่าเฉลี่ยของช่วง
        elif 'ຍາກ' in value:
            return 1  # ค่าเฉลี่ยของช่วง
        elif 'ງ່າຍ' in value:
            return 3  # ค่าเฉลี่ยของช่วง
        elif 'ງ່າຍເກີນໄປ' in value:
            return 4  # ค่าเฉลี่ยของช่วง
        else:
            try:
                return int(value)  # แปลงเป็นตัวเลขถ้าเป็นไปได้
            except ValueError:
                return value  # คงค่าเดิมถ้าแปลงไม่ได้
    return value  # คงค่าเดิมถ้าไม่ใช่ข้อความ

# ใช้ฟังก์ชันกับคอลัมน์
df['Course content difficulty'] = df['Course content difficulty'].apply(clean_expenses)

# ดูข้อมูลหลังแก้ไข
print("\nข้อมูลหลังแก้ไข:")
print(df['Course content difficulty'].head(10))

# บันทึกกลับไปที่ไฟล์ Excel (ถ้าต้องการ)
df.to_csv('Full data set 515 - ชีต1.csv', index=False)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Change df to df1 to apply encoding to the correct DataFrame
#The column name was changed in the rename operation earlier. Use the new name.
df['Main family occupation'] = le.fit_transform(df['Main family occupation']) # Changed 'ລາຍໄດ້ຫຼັກຂອງຄອບຄົວທ່ານມາຈາກອາຊີບໃດ?' to 'Main family occupation'
df['Familys main expenses'] = le.fit_transform(df['Familys main expenses']) # Changed 'ໃນຄອບຄົວເຈົ້າໃຊ້ຈ່າຍໃນເລື່ອງໃດຫຼາຍທີ່ສຸດ?' to 'Familys main expenses'
df['Final decision to drop out'] = le.fit_transform(df['Final decision to drop out'])
df['Education level'] = le.fit_transform(df['Education level'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Family income']= le.fit_transform(df['Family income'])
df['Monthly education expenses']=le.fit_transform(df['Monthly education expenses'])
# Create Economic Pressure Index
df['Economic Pressure Index'] = df['Monthly education expenses'] / df['Family income']

# Create Engagement Score
df['Engagement Score'] = (df['Satisfied with the curriculum'] +
                          df['Course content matching interests'] -
                          df['Course content difficulty'])
# Create High Absence Flag
df['High Absence Flag'] = (df['Absences per month'] > 5).astype(int)

# create Absence Category
bins = [0, 5, 15, float('inf')]
labels = ['Low', 'Medium', 'High']
df['Absence Category'] = pd.cut(df['Absences per month'], bins=bins, labels=labels, include_lowest=True)# Create Economic Pressure Index
le = LabelEncoder()
df['Absence Category'] = le.fit_transform(df['Absence Category'])
#2. ຄຳນວນຄ່າສະເລ່ຍໃນ Q1-Q35 ເພື່ອສ້າງ Feature ໃຫມ່
df['economie_factor'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)
df['school_environment_factor'] = df[['Q6', 'Q7', 'Q8', 'Q9', 'Q10']].mean(axis=1)
df['motivational_factors'] = df[['Q11', 'Q12', 'Q13', 'Q14', 'Q15']].mean(axis=1)
df['distance_factor'] = df[['Q16', 'Q17', 'Q18', 'Q19', 'Q20']].mean(axis=1)
df['health_factor'] = df[['Q21', 'Q22', 'Q23', 'Q24', 'Q25']].mean(axis=1)
df['social_factors'] = df[['Q26', 'Q27', 'Q28', 'Q29', 'Q30']].mean(axis=1)
df['family_factors'] = df[['Q31', 'Q32', 'Q33', 'Q34', 'Q35']].mean(axis=1)

# หัวข้อแอป
st.title("แอปทำนายการดรอปเรียน")

# โหลดข้อมูล
data = pd.read_csv("/content/Full data set 515 - ชีต1.csv")  # เปลี่ยนชื่อไฟล์ถ้าชื่อต่าง

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

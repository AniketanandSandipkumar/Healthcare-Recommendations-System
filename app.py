import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# Connect to your SQLite DB
conn = sqlite3.connect("project.db", check_same_thread=False)
cursor = conn.cursor()
# Prediction logs
cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    disease TEXT,
    drug TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

# Activity logs
cursor.execute("""
CREATE TABLE IF NOT EXISTS activity_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    action_type TEXT,
    details TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

# Feedback logs
cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    disease TEXT,
    sentiment TEXT,
    text TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

st.title("📊 Healthcare Recommendation System - Analytics Dashboard")

# Load Data
predictions = pd.read_sql("SELECT * FROM prediction_logs", conn)
activities = pd.read_sql("SELECT * FROM activity_logs", conn)
feedbacks = pd.read_sql("SELECT * FROM feedback_logs", conn)

st.sidebar.title("🫀 Heart Disease Prediction")

with st.sidebar.form("heart_form"):
    age = st.number_input("Age", 20, 100, 40)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
    chest_pain = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    blood_pressure = st.number_input("Resting BP", 80, 200, 120)
    cholestrol = st.number_input("Cholestrol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120 (1 = Yes, 0 = No)", [0,1])
    restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
    max_heart_rate = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina (1=Yes, 0=No)", [0,1])
    oldpeak = st.number_input("Oldpeak", -2.0, 7.0, 1.0)
    slope = st.selectbox("Slope (0-2)", [0,1,2])
    major_vessels = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
    thal = st.selectbox("Thal (0-3)", [0,1,2,3])

    submit = st.form_submit_button("Predict")

if submit:
    payload = {
        "age": age, "sex": sex, "chest_pain": chest_pain, 
        "blood_pressure": blood_pressure, "cholestrol": cholestrol,
        "fbs": fbs, "restecg": restecg, "max_heart_rate": max_heart_rate,
        "exang": exang, "oldpeak": oldpeak, "slope": slope,
        "major_vessels": major_vessels, "thal": thal
    }
    
    res = requests.post(f"{API_URL}/predict_heart", json=payload).json()
    
    st.write("### 🩺 Prediction Result:")
    st.write(f"**Heart Disease Risk:** {'YES' if res['prediction'] == 1 else 'NO'}")
    st.write(f"**Probabilities:** {res['probabilities']}")

# Existing Matplotlib/Seaborn Visuals

# Disease Distribution
st.subheader("Disease Distribution")
disease_counts = predictions["disease"].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=disease_counts.index, y=disease_counts.values, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# Drug Distribution
st.subheader("Drug Recommendation Distribution")
drug_counts = predictions["drug"].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=drug_counts.index, y=drug_counts.values, ax=ax)
st.pyplot(fig)

# User Activity Trends
st.subheader("User Activity Trends")
activity_counts = activities["action_type"].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=activity_counts.index, y=activity_counts.values, ax=ax)
st.pyplot(fig)

# Feedback Sentiment
st.subheader("Feedback Sentiment Analysis")
sentiment_counts = feedbacks["sentiment"].value_counts()
fig, ax = plt.subplots()
plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct="%1.1f%%")
st.pyplot(fig)


# Power BI Dashboards Integration


st.subheader("📈 Power BI Dashboards")

# Use tabs to display both dashboards side by side
tab1, tab2 = st.tabs(["Patient Risk & Demographics", "Symptoms & Feature / Exercise Risk"])

with tab1:
    st.markdown("**Patient Risk & Demographics Dashboard**")
    powerbi_link1 = "https://app.powerbi.com/reportEmbed?reportId=26314451-b947-4c3a-a525-fbcff2f06ba7&autoAuth=true&ctid=b10b7583-c2ed-4f35-8815-ed38d24ed1be"  # Replace with your published Power BI link
    st.components.v1.html(f"""
        <iframe width="100%" height="600" src="{powerbi_link1}" frameborder="0" allowFullScreen="true"></iframe>
    """, height=620)

with tab2:
    st.markdown("**Symptoms & Feature / Exercise Risk Dashboard**")
    powerbi_link2 = "YOUR_POWERBI_EMBED_LINK_2"  # Replace with your published Power BI link
    st.components.v1.html(f"""
        <iframe width="100%" height="600" src="{powerbi_link2}" frameborder="0" allowFullScreen="true"></iframe>
    """, height=620)



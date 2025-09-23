import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import requests

API_URL = "https://healthcare-backend-5omj.onrender.com"  # Replace with Render backend URL

st.title("📊 Healthcare Recommendation System - Analytics Dashboard")

# ----------------- FUNCTIONS -----------------
def get_db_connection():
    conn = sqlite3.connect("./app.db", check_same_thread=False)
    return conn

def load_predictions():
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT * FROM logs", conn)
    except Exception:
        # Return empty DataFrame if table does not exist
        df = pd.DataFrame(columns=["id", "user_id", "disease", "drug", "timestamp"])
    conn.close()
    return df

def load_activities():
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT * FROM activities", conn)
    except Exception:
        df = pd.DataFrame(columns=["id", "user_id", "action_type", "details", "timestamp"])
    conn.close()
    return df

def load_feedbacks():
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT * FROM feedback", conn)
    except Exception:
        df = pd.DataFrame(columns=["id", "user_id", "prediction_id", "text", "sentiment", "timestamp"])
    conn.close()
    return df


# ----------------- SIDEBAR HEART FORM -----------------
with st.sidebar.form("heart_form"):
    age = st.number_input("Age", 20, 100, 40)
    sex = st.selectbox("Sex (0=Female,1=Male)", [0,1])
    chest_pain = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    blood_pressure = st.number_input("Resting BP", 80, 200, 120)
    cholestrol = st.number_input("Cholestrol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
    restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
    max_heart_rate = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [0,1])
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
    st.write(f"**Heart Disease Risk:** {'YES' if res['prediction']==1 else 'NO'}")
    st.write(f"**Probabilities:** {res['probabilities']}")

# ----------------- DASHBOARD -----------------
predictions = load_predictions()
activities = load_activities()
feedbacks = load_feedbacks()

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

# Power BI Dashboards (iframe)
st.subheader("📈 Power BI Dashboards")
tab1, tab2 = st.tabs(["Patient Risk & Demographics", "Symptoms & Feature / Exercise Risk"])
with tab1:
    powerbi_link1 = "YOUR_POWERBI_LINK_1"
    st.components.v1.html(f'<iframe width="100%" height="600" src="{https://app.powerbi.com/reportEmbed?reportId=26314451-b947-4c3a-a525-fbcff2f06ba7&autoAuth=true&ctid=b10b7583-c2ed-4f35-8815-ed38d24ed1be}" frameborder="0" allowFullScreen="true"></iframe>', height=620)
with tab2:
    powerbi_link2 = "YOUR_POWERBI_LINK_2"
    st.components.v1.html(f'<iframe width="100%" height="600" src="{https://app.powerbi.com/reportEmbed?reportId=26314451-b947-4c3a-a525-fbcff2f06ba7&autoAuth=true&ctid=b10b7583-c2ed-4f35-8815-ed38d24ed1be}" frameborder="0" allowFullScreen="true"></iframe>', height=620)








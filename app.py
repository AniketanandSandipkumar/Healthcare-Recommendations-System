import streamlit as st
import requests
import pandas as pd

API_URL = "https://healthcare-backend.onrender.com"  # Replace with actual backend URL

st.set_page_config(page_title="Healthcare Recommender", layout="wide")
st.title("🏥 Healthcare Recommendation System")

# ============= Heart Disease Form =============
st.sidebar.header("Heart Disease Prediction")
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
    payload = {"age": age,"sex": sex,"chest_pain": chest_pain,
        "blood_pressure": blood_pressure,"cholestrol": cholestrol,
        "fbs": fbs,"restecg": restecg,"max_heart_rate": max_heart_rate,
        "exang": exang,"oldpeak": oldpeak,"slope": slope,
        "major_vessels": major_vessels,"thal": thal}
    try:
        res = requests.post(f"{API_URL}/predict_heart", json=payload).json()
        st.write("### 🩺 Prediction Result:")
        st.write(f"**Heart Disease Risk:** {'YES' if res['prediction']==1 else 'NO'}")
        st.write(f"**Probabilities:** {res['probabilities']}")
    except Exception as e:
        st.error(f"Error: {e}")

# ============= Disease-Drug Recommendation =============
st.subheader("💊 Disease & Drug Recommendation")
with st.form("disease_form"):
    disease_name = st.text_input("Enter Disease Name")
    submit_disease = st.form_submit_button("Get Recommendations")

if submit_disease and disease_name.strip() != "":
    try:
        res = requests.get(f"{API_URL}/recommend_knn/{disease_name}").json()
        if "recommendations" in res:
            st.dataframe(pd.DataFrame(res["recommendations"]))
        else:
            st.warning(res.get("detail", "⚠️ No recommendations found."))
    except Exception as e:
        st.error(f"Error fetching recommendations: {e}")

# ============= Analytics + Power BI =============
st.markdown("---")
st.subheader("📊 Interactive Analytics")
import analytics  # renders all Plotly charts

# Power BI Dashboards (iframe)
st.subheader("📈 Power BI Dashboards")
tab1, tab2 = st.tabs(["Patient Risk & Demographics", "Symptoms & Feature / Exercise Risk"])
with tab1:
    powerbi_link1 = "https://app.powerbi.com/reportEmbed?reportId=26314451-b947-4c3a-a525-fbcff2f06ba7&autoAuth=true&ctid=b10b7583-c2ed-4f35-8815-ed38d24ed1be"
    st.components.v1.html(f'<iframe width="100%" height="600" src="{powerbi_link1}" frameborder="0" allowFullScreen="true"></iframe>', height=620)
with tab2:
    powerbi_link2 = "https://app.powerbi.com/reportEmbed?reportId=26314451-b947-4c3a-a525-fbcff2f06ba7&autoAuth=true&ctid=b10b7583-c2ed-4f35-8815-ed38d24ed1be"
    st.components.v1.html(f'<iframe width="100%" height="600" src="{powerbi_link2}" frameborder="0" allowFullScreen="true"></iframe>', height=620)












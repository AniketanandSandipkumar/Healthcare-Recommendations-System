import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
# ----------------- LOAD MODELS & DATA -----------------
heart_model = joblib.load("heart_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

df_knn = pd.read_csv("disease_drug_mapping.csv")
symptom_cols = [c for c in df_knn.columns if "Symptom" in c]
feature_matrix = scaler.transform(df_knn[symptom_cols])

df_heart = pd.read_csv("heart_cleveland_clean.csv")

# ----------------- FUNCTIONS -----------------
def predict_heart(features):
    X = pd.DataFrame([features], columns=[
        "age","sex","chest_pain","blood_pressure","cholestrol",
        "fbs","restecg","max_heart_rate","exang",
        "oldpeak","slope","major_vessels","thal"
    ])
    X_scaled = heart_scaler.transform(X)
    prediction = heart_model.predict(X_scaled)[0]
    proba = heart_model.predict_proba(X_scaled)[0].tolist()
    return prediction, proba

def get_knn_recommendations(disease_name, num_recs=5):
    df_knn['Disease'] = df_knn['Disease'].str.strip().str.lower()
    disease_name = disease_name.strip().lower()
    matches = df_knn[df_knn['Disease'].str.contains(disease_name, case=False, na=False)]['Disease'].unique()
    if len(matches) == 0: return None
    closest_match = matches[0]
    idx = df_knn[df_knn['Disease'] == closest_match].index[0]
    distances, indices = knn_model.kneighbors([feature_matrix[idx]])
    recommended_diseases = df_knn.iloc[indices[0][1:num_recs+1]]
    return recommended_diseases[['Disease','Drug']]
    
def log_activity(user_id, event_type, item):
    """Send activity to backend"""
    try:
        requests.post(
            f"{BACKEND_URL}/activity/log",
            json={"user_id": user_id, "event_type": event_type, "item": item},
            timeout=2
        )
    except Exception as e:
        print("Activity logging failed:", e)
        
# ----------------- STREAMLIT UI -----------------
st.title("🏥 Healthcare Recommendation System")

# ---------- Authentication ----------
st.sidebar.header("Login / Signup")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
if st.sidebar.button("Login"):
    st.session_state["user"] = username
    st.success(f"Welcome {username}!")

# ---------- Heart Disease Prediction ----------
st.subheader("❤️ Heart Disease Prediction")
with st.form("heart_form"):
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
    features = [age, sex, chest_pain, blood_pressure, cholestrol,
                fbs, restecg, max_heart_rate, exang,
                oldpeak, slope, major_vessels, thal]
    pred, proba = predict_heart(features)
    st.write("### 🩺 Prediction Result:")
    st.write(f"**Heart Disease Risk:** {'YES' if pred==1 else 'NO'}")
    st.write(f"**Probabilities:** {{'No Disease': {proba[0]:.2f}, 'Disease': {proba[1]:.2f}}}")

# ---------- Disease & Drug Recommendation ----------
st.subheader("💊 Disease & Drug Recommendation")
with st.form("disease_form"):
    disease_name = st.text_input("Enter Disease Name", "")
    submit_disease = st.form_submit_button("Get Recommendations")

if submit_disease and disease_name.strip() != "":
    recs = get_knn_recommendations(disease_name)
    if recs is None:
        st.warning(f"⚠️ No recommendations found for '{disease_name}'.")
    else:
        st.write("### Recommended Diseases & Drugs")
        st.dataframe(recs)
        
 # Log activity
        if "user_id" in st.session_state:
            log_activity(st.session_state["user_id"], "disease_lookup", disease_name)

# ---------- PLOTLY VISUALIZATIONS ----------
st.subheader("📊 Plotly Insights")

# 1. Heart Disease Count by Age
fig1 = px.histogram(df_heart, x="age", nbins=20, color="target",
                    labels={"target":"Heart Disease"}, 
                    title="Heart Disease Count by Age in Cleveland Dataset")
st.plotly_chart(fig1)

# 2. Heart Disease Probability by Chest Pain Type
fig2 = px.bar(df_heart.groupby("cp")["target"].mean().reset_index(),
              x="cp", y="target", labels={"cp":"Chest Pain Type", "target":"Disease Probability"},
              title="Heart Disease Probability by Chest Pain Type")
st.plotly_chart(fig2)

# 3. Heart Disease Distribution by Sex
fig3 = px.pie(df_heart, names="sex", values="target", 
              title="Heart Disease Distribution by Sex")
st.plotly_chart(fig3)

# 4. Correlation Heatmap (Heart Features)
corr = df_heart.corr()
fig4 = px.imshow(corr, text_auto=True, aspect="auto", 
                 title="Feature Correlation Heatmap (Heart Cleveland Dataset)")
st.plotly_chart(fig4)

# 5. Top 10 Diseases in Disease-Drug Dataset
top_diseases = df_knn['Disease'].value_counts().nlargest(10).reset_index()
top_diseases.columns = ["Disease", "Count"]
fig5 = px.bar(top_diseases, x="Disease", y="Count", 
              title="Top 10 Diseases in Disease-Drug Mapping")
st.plotly_chart(fig5)

# 6. Top 10 Drugs in Disease-Drug Dataset
top_drugs = df_knn['Drug'].value_counts().nlargest(10).reset_index()
top_drugs.columns = ["Drug","Count"]
fig6 = px.bar(top_drugs, x="Drug", y="Count", 
              title="Top 10 Drugs in Disease-Drug Mapping")
st.plotly_chart(fig6)

# 7. Max Heart Rate vs Resting BP by Heart Disease
fig7 = px.scatter(df_heart, x="trestbps", y="thalach", color="target",
                  labels={"trestbps":"Resting Blood Pressure", "thalach":"Max Heart Rate", "target":"Heart Disease"},
                  title="Max Heart Rate vs Resting BP colored by Heart Disease")
st.plotly_chart(fig7)

# 8. Cholestrol vs Age colored by Heart Disease
fig8 = px.scatter(df_heart, x="age", y="chol", color="target",
                  labels={"chol":"Cholestrol", "target":"Heart Disease"},
                  title="Cholestrol vs Age with Heart Disease")
st.plotly_chart(fig8)

# 9. Top 10 Symptoms in Disease-Drug Mapping
symptom_counts = df_knn[symptom_cols].sum().sort_values(ascending=False).nlargest(10).reset_index()
symptom_counts.columns = ["Symptom","Count"]
fig9 = px.bar(symptom_counts, x="Symptom", y="Count", 
              title="Top 10 Symptoms in Disease-Drug Mapping")
st.plotly_chart(fig9)

# 10. Disease-Drug Relationship Heatmap
drug_disease_matrix = pd.crosstab(df_knn['Disease'], df_knn['Drug'])
fig10 = px.imshow(drug_disease_matrix, text_auto=True, aspect="auto", 
                  labels={"x":"Drug", "y":"Disease", "color":"Count"},
                  title="Disease-Drug Relationship Heatmap")
st.plotly_chart(fig10)

# ---------- POWER BI DASHBOARDS ----------
st.subheader("📈 Power BI Dashboards")

# Copy-paste links here
POWERBI_LINK_1 = "https://app.powerbi.com/reportEmbed?reportId=26314451-b947-4c3a-a525-fbcff2f06ba7&autoAuth=true&ctid=b10b7583-c2ed-4f35-8815-ed38d24ed1be"
POWERBI_LINK_2 = "https://app.powerbi.com/reportEmbed?reportId=26314451-b947-4c3a-a525-fbcff2f06ba7&autoAuth=true&ctid=b10b7583-c2ed-4f35-8815-ed38d24ed1be"

tab1, tab2 = st.tabs(["Patient Risk & Demographics", "Symptoms & Feature / Exercise Risk"])
with tab1:
    st.components.v1.html(f'<iframe width="100%" height="600" src="{POWERBI_LINK_1}" frameborder="0" allowFullScreen="true"></iframe>', height=620)
with tab2:
    st.components.v1.html(f'<iframe width="100%" height="600" src="{POWERBI_LINK_2}" frameborder="0" allowFullScreen="true"></iframe>', height=620)


st.title("📊 Real-Time User Activity Dashboard")
st.write("This dashboard displays all user interactions logged by the backend.")
# Replace this with your deployed Render backend debug route
BACKEND_URL = "https://your-backend.onrender.com/debug/activities"

try:
    response = requests.get(BACKEND_URL)
    if response.status_code == 200:
        activities = response.json()
    else:
        st.error(f"Failed to fetch data. Status code: {response.status_code}")
        activities = []
except Exception as e:
    st.error(f"Error fetching data: {e}")
    activities = []
if activities:
    # Convert timestamp to datetime
    for a in activities:
        a['timestamp'] = datetime.fromisoformat(a['timestamp'])

    df = pd.DataFrame(activities)
    st.dataframe(df)
else:
    st.write("No user activity logged yet.")


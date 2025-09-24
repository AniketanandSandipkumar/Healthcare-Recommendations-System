import streamlit as st
import pandas as pd
import joblib

# ----------------- LOAD MODELS -----------------
heart_model = joblib.load("heart_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

df_knn = pd.read_csv("disease_drug_mapping.csv")
symptom_cols = [c for c in df_knn.columns if "Symptom" in c]
feature_matrix = scaler.transform(df_knn[symptom_cols])

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

# ----------------- STREAMLIT UI -----------------
st.title("🏥 Healthcare Recommendation System")

# ----------- Authentication (Frontend only check) -----------
st.sidebar.header("Login / Signup")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
if st.sidebar.button("Login"):
    st.session_state["user"] = username
    st.success(f"Welcome {username}!")

# ----------- Heart Disease Prediction -----------
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

# ----------- Disease & Drug Recommendation -----------
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

# 🏥 Healthcare Recommendation System

An end-to-end **Personalized Healthcare Recommendation System** that leverages **Machine Learning, Data Analytics, and Interactive Dashboards** to predict heart disease risk, recommend diseases and drugs, and analyze user behavior.

This project integrates:
- ML-powered medical predictions  
- KNN-based disease & drug recommendations  
- Interactive Streamlit UI  
- Secure FastAPI backend with JWT authentication  
- Behavioral tracking & analytics  
- Plotly & Power BI dashboards  
- Cloud deployment using Render  

---

## 📌 Project Overview

The **Healthcare Recommendation System** is designed to assist users and healthcare analysts by:

- Predicting **heart disease risk**
- Recommending **related diseases and medicines**
- Tracking **user interactions and behavior**
- Visualizing insights through **interactive dashboards**

The system is modular, scalable, and deployable, making it suitable for real-world healthcare analytics as well as academic and portfolio projects.

---

## 🧠 Key Features

### 🔐 User Authentication & Management
- Secure **Signup/Login** using JWT authentication  
- Password hashing with `bcrypt`  
- Role-based access (User / Admin / Analyst)  
- Backend validation using FastAPI  

---

### ❤️ Heart Disease Prediction
- Machine Learning classifier trained on the **Cleveland Heart Disease Dataset**
- Input parameters include:
  - Age, Blood Pressure, Cholesterol
  - ECG results, Heart Rate, Chest Pain type, etc.
- Outputs:
  - Disease prediction (Yes / No)
  - Probability score

**Model & Dataset Files**
- `heart_model.pkl`
- `heart_scaler.pkl`
- `heart_cleveland_clean.csv`

---

### 💊 Disease & Drug Recommendation System
- **KNN-based recommendation engine**
- Suggests similar diseases and recommended drugs
- Supports partial and approximate disease name matching
- Uses symptom-based feature similarity

**Model & Dataset Files**
- `knn_model.pkl`
- `scaler.pkl`
- `disease_drug_mapping.csv`

---

### 📊 Analytics & Visualization
- **Plotly-based interactive charts**, including:
  - Heart disease vs age
  - Disease probability by chest pain type
  - Feature correlation heatmap
  - Top diseases and drugs
  - Symptom frequency analysis
- **Power BI dashboards**
  - Embedded into Streamlit via iframe
  - Advanced analytical views for admins and analysts

---

### 🧾 User Behavior Tracking
- Tracks:
  - Heart disease prediction requests
  - Disease and drug recommendation interactions
- Stored using the `UserActivity` database table
- Enables behavioral analytics and future personalization

---

## 🏗️ System Architecture
User<br>
└── Streamlit Frontend (app.py)<br>
├── Heart Disease Prediction (ML)<br>
├── Disease & Drug Recommendation (KNN)<br>
├── Plotly Visualizations<br>
└── Power BI Dashboard (Embedded)<br>
↓
FastAPI Backend (main.py)<br><br>
├── JWT Authentication<br>
├── User Management<br>
└── Activity Logging<br>
↓
SQLite Database (db.py)<br>
---

---

## 🗂️ Project Structure

Healthcare-Recommendations-System/<br>
│
├── app.py # Streamlit frontend<br>
├── main.py # FastAPI backend<br>
├── db.py # Database models & engine<br>
├── analytics.py # Analytics helpers<br>
│
├── heart_model.pkl # Trained heart disease model<br>
├── heart_scaler.pkl # Feature scaler<br>
├── knn_model.pkl # KNN recommendation model<br>
├── scaler.pkl # Symptom scaler<br>
│
├── heart_cleveland_clean.csv # Heart disease dataset<br>
├── disease_drug_mapping.csv # Disease-drug mapping dataset<br>
│
├── requirements.txt # Python dependencies<br>
├── render.yaml # Render deployment config<br>
├── README.md # Project documentation<br>
└── .gitignore<br>


---

## ⚙️ Tech Stack

| Category       | Technologies                          |
|----------------|---------------------------------------|
| Frontend       | Streamlit, Plotly                     |
| Backend        | FastAPI, SQLAlchemy                   |
| Machine Learning | Random Forest, KNN                  |
| Authentication | JWT, OAuth2                           |
| Database       | SQLite                                |
| Dashboards     | Power BI                              |
| Deployment     | Render                                |
| Language       | Python                                |

---

## 🚀 Deployment

###***Backend (FastAPI)
- Hosted on Render
- Configured using `render.yaml`
- Startup command:
uvicorn main:app --host 0.0.0.0 --port 10000

##***Frontend (Streamlit)<br>

-Uses trained ML models for predictions<br>
-Connects to backend APIs for authentication and activity tracking<br>

##📈 Future Enhancements<br>

1.Collaborative filtering recommendations<br>
2.Sentiment analysis on user feedback<br>
3.Reinforcement learning-based adaptive recommendations<br>
4.Migration to PostgreSQL<br>
5.Role-based analytical dashboards<br>
6.Explainable AI (XAI) for medical predictions<br>

👨‍💻 Author<br>
Aniket Anand<br>
B.Tech Computer Science<br>
AI | Machine Learning | Data Science | Full Stack Development<br>
GitHub: AniketanandSandipkumar<br>
App link:https://healthcare-frontend-rfkp.onrender.com (frontend)<br>
         https://healthcare-backend-5omj.onrender.com (backend)<br><br>
         
⭐ Acknowledgements<br>
*UCI Cleveland Heart Disease Dataset
*Streamlit & FastAPI developer communities
*Power BI analytics tools

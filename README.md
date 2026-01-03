🏥 Healthcare Recommendation System

An end-to-end Personalized Healthcare Recommendation System that leverages Machine Learning, Data Analytics, and Interactive Dashboards to predict heart disease risk, recommend diseases and drugs, and analyze user behavior.

This project integrates:
-ML-powered medical predictions
-KNN-based disease & drug recommendations
-Interactive Streamlit UI
-Secure FastAPI backend with JWT authentication
-Behavioral tracking & analytics
-Plotly & Power BI dashboards
-Cloud deployment using Render

📌 Project Overview

The Healthcare Recommendation System is designed to assist users and healthcare analysts by:
-Predicting heart disease risk
-Recommending related diseases and medicines
-Tracking user interactions and behavior
-Visualizing insights through interactive dashboards
-The system is modular, scalable, and deployable, making it suitable for real-world healthcare analytics and academic projects.

🧠 Key Features
🔐 User Authentication & Management

-Secure Signup/Login using JWT authentication
-Password hashing with bcrypt
-Role support (User / Admin / Analyst)
-Backend validation using FastAPI

❤️ Heart Disease Prediction

-Machine Learning classifier trained on Cleveland Heart Disease Dataset
Inputs include:
Age, Blood Pressure, Cholesterol, ECG, Heart Rate, etc.

Outputs:
Disease prediction (Yes / No)
Probability scores

Model Files Used:
heart_model.pkl
heart_scaler.pkl

Dataset: heart_cleveland_clean.csv

💊 Disease & Drug Recommendation System

-KNN-based Recommendation Engine
-Suggests similar diseases and recommended drugs
-Handles partial and approximate disease names
-Uses symptom-based feature similarity

Model & Data
-knn_model.pkl
-scaler.pkl

Dataset: disease_drug_mapping.csv

📊 Analytics & Visualization

Plotly Visualizations (10+) including:

Heart disease vs age

Disease probability by chest pain type

Feature correlation heatmap

Top diseases & drugs

Symptom frequency analysis

Power BI Dashboards

Embedded into Streamlit using iframe links

Provides advanced insights for analysts and admins

🧾 User Behavior Tracking

Tracks:

Clicks on diseases

Drug recommendations

Heart disease predictions

Stored in database using UserActivity table

Enables behavioral analytics and future personalization

🏗️ System Architecture
User
 └── Streamlit Frontend (app.py)
      ├── Heart Disease Prediction (ML)
      ├── Disease & Drug Recommendation (KNN)
      ├── Plotly Visualizations
      └── Power BI Dashboard (Embedded)
               ↓
        FastAPI Backend (main.py)
      ├── JWT Authentication
      ├── User Management
      └── Activity Logging
               ↓
          SQLite Database (db.py)

🗂️ Project Structure
Healthcare-Recommendations-System/
│
├── app.py                     # Streamlit frontend
├── main.py                    # FastAPI backend
├── db.py                      # Database models & engine
├── analytics.py               # Analytics helpers
│
├── heart_model.pkl            # Trained heart disease model
├── heart_scaler.pkl           # Feature scaler for heart model
├── knn_model.pkl              # KNN recommendation model
├── scaler.pkl                 # Symptom scaler
│
├── heart_cleveland_clean.csv  # Heart disease dataset
├── disease_drug_mapping.csv   # Disease-drug mapping dataset
│
├── requirements.txt           # Python dependencies
├── render.yaml                # Render deployment config
├── README.md                  # Project documentation
└── .gitignore

⚙️ Tech Stack
Category	Technologies
Frontend	Streamlit, Plotly
Backend	FastAPI, SQLAlchemy
ML Models	Random Forest / Classifier, KNN
Auth	JWT, OAuth2
Database	SQLite
Dashboards	Power BI
Deployment	Render
Language	Python
🚀 Deployment
Backend (FastAPI)

Hosted on Render

Uses render.yaml

Entry command:

uvicorn main:app --host 0.0.0.0 --port 10000

Frontend (Streamlit)

Uses models directly for predictions

Connects to backend for authentication & activity tracking

📈 Future Enhancements

Collaborative filtering recommendations

Sentiment analysis on user feedback

Reinforcement learning for adaptive recommendations

Migration to PostgreSQL

Role-based dashboards

Explainable AI (XAI) for medical predictions

👨‍💻 Author

Aniket Anand
B.Tech Computer Science
AI | ML | Data Science | Full Stack
🔗 GitHub: AniketanandSandipkumar

⭐ Acknowledgements

UCI Cleveland Heart Disease Dataset

Streamlit & FastAPI communities

Power BI analytics tools

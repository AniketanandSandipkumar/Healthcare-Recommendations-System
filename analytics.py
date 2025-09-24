# analytics.py
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

DB_PATH = "app.db"

def safe_read_sql(query, conn, empty_cols):
    """Try to read SQL, return empty DataFrame if table missing."""
    try:
        return pd.read_sql(query, conn)
    except Exception:
        return pd.DataFrame(columns=empty_cols)

# Connect
conn = sqlite3.connect(DB_PATH)

@st.cache_data
def load_data():
    predictions = safe_read_sql("SELECT * FROM logs", conn, ["id","user_id","disease","drug","timestamp"])
    feedback = safe_read_sql("SELECT * FROM feedback", conn, ["id","user_id","prediction_id","text","sentiment","timestamp"])
    users = safe_read_sql("SELECT * FROM users", conn, ["id","username","role","age","gender","preferences"])
    return predictions, feedback, users

predictions, feedback, users = load_data()

st.markdown("### 📊 Analytics Dashboard")

# --- Disease Frequency ---
if not predictions.empty:
    top_diseases = predictions['disease'].value_counts().head(10)
    st.plotly_chart(px.bar(top_diseases, x=top_diseases.index, y=top_diseases.values, title="Top 10 Predicted Diseases"))
else:
    st.info("No predictions logged yet.")

# --- Drug Frequency ---
if not predictions.empty:
    top_drugs = predictions['drug'].value_counts().head(10)
    st.plotly_chart(px.bar(top_drugs, x=top_drugs.index, y=top_drugs.values, title="Top 10 Recommended Drugs"))

# --- Sentiment ---
if not feedback.empty:
    sentiment_counts = feedback['sentiment'].value_counts()
    st.plotly_chart(px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values, title="Feedback Sentiment"))

# --- User Roles ---
if not users.empty:
    role_counts = users['role'].value_counts()
    st.plotly_chart(px.pie(role_counts, names=role_counts.index, values=role_counts.values, hole=0.4, title="User Role Distribution"))

# --- Word Cloud ---
if not feedback.empty:
    st.subheader("Most Common Feedback Words")
    all_text = " ".join(feedback['text'].dropna().tolist())
    if all_text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        plt.imshow(wordcloud, interpolation='bilinear'); plt.axis('off')
        st.pyplot(plt)


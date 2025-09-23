import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# Connect to your SQLite DB
conn = sqlite3.connect("app.db")

st.title("📊 Healthcare Recommendation System - Analytics Dashboard")

# Load Data
predictions = pd.read_sql("SELECT * FROM logs", conn)
activities = pd.read_sql("SELECT * FROM activities", conn)
feedbacks = pd.read_sql("SELECT * FROM feedback", conn)

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

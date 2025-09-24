import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

conn = sqlite3.connect('app.db')

@st.cache_data
def load_data():
    predictions = pd.read_sql("SELECT * FROM logs", conn)
    feedback = pd.read_sql("SELECT * FROM feedback", conn)
    users = pd.read_sql("SELECT * FROM users", conn)
    return predictions, feedback, users

predictions, feedback, users = load_data()
st.markdown("### 📊 Analytics Dashboard")

# Disease Frequency
top_diseases = predictions['disease'].value_counts().head(10)
st.plotly_chart(px.bar(top_diseases, x=top_diseases.index, y=top_diseases.values, title="Top 10 Predicted Diseases"))

# Drug Frequency
top_drugs = predictions['drug'].value_counts().head(10)
st.plotly_chart(px.bar(top_drugs, x=top_drugs.index, y=top_drugs.values, title="Top 10 Recommended Drugs"))

# Sentiment
sentiment_counts = feedback['sentiment'].value_counts()
st.plotly_chart(px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values, title="Feedback Sentiment"))

# User Roles
role_counts = users['role'].value_counts()
st.plotly_chart(px.pie(role_counts, names=role_counts.index, values=role_counts.values, hole=0.4, title="User Role Distribution"))

# Word Cloud
st.subheader("Most Common Feedback Words")
all_text = " ".join(feedback['text'].dropna().tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.imshow(wordcloud, interpolation='bilinear'); plt.axis('off')
st.pyplot(plt)

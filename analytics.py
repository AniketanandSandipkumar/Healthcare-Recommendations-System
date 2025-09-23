# analytics.py
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# 1. Connect to SQLite Database
conn = sqlite3.connect('healthcare.db')  # Update with your DB path

# 2. Load Data
@st.cache_data
def load_data():
    predictions = pd.read_sql("SELECT * FROM predictions", conn)
    feedback = pd.read_sql("SELECT * FROM feedback", conn)
    users = pd.read_sql("SELECT * FROM users", conn)
    return predictions, feedback, users

predictions, feedback, users = load_data()

st.title("Healthcare Recommendation System Analytics Dashboard")

# 3. Disease Frequency Bar Chart
st.subheader("Top 10 Predicted Diseases")
top_diseases = predictions['disease'].value_counts().head(10)
fig_disease = px.bar(top_diseases, x=top_diseases.index, y=top_diseases.values, 
                     labels={'x':'Disease', 'y':'Count'}, title="Top 10 Predicted Diseases")
st.plotly_chart(fig_disease)


# 4. Drug Recommendation Bar Chart
st.subheader("Top 10 Recommended Drugs")
top_drugs = predictions['drug'].value_counts().head(10)
fig_drug = px.bar(top_drugs, x=top_drugs.index, y=top_drugs.values,
                  labels={'x':'Drug', 'y':'Count'}, title="Top 10 Recommended Drugs")
st.plotly_chart(fig_drug)

# 5. Disease vs Drug Heatmap
st.subheader("Disease vs Drug Heatmap")
disease_drug_matrix = pd.crosstab(predictions['disease'], predictions['drug'])
fig_heatmap = px.imshow(disease_drug_matrix, labels=dict(x="Drug", y="Disease", color="Count"),
                        title="Disease vs Drug Correlation")
st.plotly_chart(fig_heatmap)

# 6. Sentiment Distribution Pie Chart
st.subheader("Feedback Sentiment Distribution")
sentiment_counts = feedback['sentiment'].value_counts()
fig_sentiment = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values,
                       title="Sentiment Distribution")
st.plotly_chart(fig_sentiment)

# 7. Time-Series Line Chart (Predictions per Day)
st.subheader("Predictions Over Time")
predictions['prediction_date'] = pd.to_datetime(predictions['prediction_date'])
time_series = predictions.groupby(predictions['prediction_date'].dt.date).size().reset_index(name='Count')
fig_time = px.line(time_series, x='prediction_date', y='Count', title="Predictions Over Time")
st.plotly_chart(fig_time)

# 8. User Activity Line Chart
st.subheader("User Activity Trends")
activity_daily = pd.merge(predictions.groupby(predictions['prediction_date'].dt.date).size().reset_index(name='Predictions'),
                          feedback.groupby(feedback['id']).size().reset_index(name='Feedback'),
                          left_index=True, right_index=True, how='outer').fillna(0)
fig_activity = px.line(activity_daily, x=activity_daily.index, y=['Predictions','Feedback'],
                       title="User Activity Trends (Predictions & Feedback)")
st.plotly_chart(fig_activity)

# 9. User Role Distribution Donut Chart
st.subheader("User Role Distribution")
role_counts = users['role'].value_counts()
fig_role = px.pie(role_counts, names=role_counts.index, values=role_counts.values, hole=0.4,
                  title="User Role Distribution")
st.plotly_chart(fig_role)


# 10. Most Common Symptoms (Word Cloud)
st.subheader("Most Common Symptoms Word Cloud")
all_feedback_text = " ".join(feedback['feedback_text'].dropna().tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_feedback_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)


# 11. Feedback Sentiment by Disease (Stacked Bar)
st.subheader("Feedback Sentiment by Disease")
feedback_grouped = feedback.groupby(['disease', 'sentiment']).size().reset_index(name='Count')
fig_feedback_disease = px.bar(feedback_grouped, x='disease', y='Count', color='sentiment',
                              title="Feedback Sentiment per Disease")
st.plotly_chart(fig_feedback_disease)


# 12. Interactive Treemap / Sunburst
st.subheader("Disease → Drug → Sentiment Hierarchy")
treemap_data = predictions.merge(feedback, on=['user_id','disease'], how='left').fillna('Neutral')
fig_treemap = px.treemap(treemap_data, path=['disease','drug','sentiment'], values=None,
                         color='sentiment', color_discrete_map={'Positive':'green','Negative':'red','Neutral':'grey'},
                         title="Disease → Drug → Sentiment Hierarchy")
st.plotly_chart(fig_treemap)

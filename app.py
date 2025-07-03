import streamlit as st
import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob


with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

st.set_page_config(page_title="Employee Sentiment Classifier", layout="centered")
st.title(" Employee Feedback Sentiment Analysis")

user_input = st.text_area(" Enter employee feedback here:")

if st.button(" Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some feedback to analyze.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        polarity = TextBlob(user_input).sentiment.polarity

        st.markdown("### Results")
        st.write(f"**Original Input:** {user_input}")
        st.write(f"**Cleaned Text:** {cleaned}")
        st.write(f"**Predicted Sentiment:** `{prediction}`")
        st.write(f"**TextBlob Polarity Score:** `{round(polarity, 3)}`")

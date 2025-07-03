import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from textblob import TextBlob

with open(r"C:\Users\Khushi Wadhawan\AppData\Roaming\Python\Python311\Scripts\sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(r"C:\Users\Khushi Wadhawan\AppData\Roaming\Python\Python311\Scripts\vectorizer.pkl", "rb") as f:
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

st.set_page_config(page_title="Smart Sentiment Dashboard", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #f0f0f0;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #90ee90;
        text-align: center;
        margin-bottom: 10px;
    }
    .about-box {
        font-size: 14px;
        background-color: #2a2a2a;
        color: #cccccc;
        padding: 10px;
        border-left: 4px solid #90ee90;
        border-radius: 6px;
        margin-bottom: 25px;
    }
    .result-box {
        background-color: #2a2a2a;
        color: #e6e6e6;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    textarea {
        background-color: #2c3e2f !important;
        color: #e2f2de !important;
        font-size: 16px !important;
        border-radius: 6px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Smart Sentiment Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
<div class="about-box">
This tool classifies employee feedback as Positive, Neutral, or Negative using ML and TextBlob.
</div>
""", unsafe_allow_html=True)

user_input = st.text_area("Enter employee feedback:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some feedback.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        polarity = TextBlob(user_input).sentiment.polarity

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        st.write(f"**Original Feedback:** {user_input}")
        st.write(f"**Cleaned Text:** {cleaned}")
        st.write(f"**Predicted Sentiment:** `{prediction}`")
        st.write(f"**Polarity Score:** `{round(polarity, 3)}`")
        st.markdown('</div>', unsafe_allow_html=True)

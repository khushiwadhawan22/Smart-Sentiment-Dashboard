# Smart Sentiment Dashboard

A lightweight and professional web application that classifies employee feedback into Positive, Neutral, or Negative sentiment using rule-based logic, TextBlob polarity, and a machine learning model.

---

## Overview

- Processes employee feedback and predicts sentiment in real-time.
- Combines rule-based keywords, TextBlob polarity, and a Logistic Regression model.
- Uses SMOTE to balance class distribution during training.
- Deployed as a clean, interactive dashboard built with Streamlit.

---

## Use Cases

- Quickly analyze survey or interview responses in HR workflows.
- Monitor sentiment trends across teams or departments.
- Identify common pain points or areas of improvement through qualitative feedback.

---

## Tech Stack

- **Python**: Core development
- **Pandas**, **NumPy**: Data manipulation
- **NLTK**, **TextBlob**: Natural Language Processing
- **Scikit-learn**: Model training (Logistic Regression + GridSearchCV)
- **Imbalanced-learn**: SMOTE for oversampling
- **Streamlit**: UI for dashboard interface
- **Matplotlib**, **Seaborn**: Visualizations (EDA notebook)
- **Git & GitHub**: Version control and deployment

---

## How to Run

```bash
streamlit run dashboard.py

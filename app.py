import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk

# NLTK downloads
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load all components
model = joblib.load("best_fake_news_model.pkl")
vectorizer = joblib.load("news_tfidf_vectorizer.pkl")
scaler = joblib.load("news_scaler.pkl")
encoder = joblib.load("news_encoder.pkl")

# --- Text Cleaning Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- Streamlit App UI ---
st.set_page_config(page_title="Fake News Detector", layout="centered", page_icon="🕵️")
st.title("🕵️‍♂️ Fake News Detection App")
st.markdown("This app predicts whether a news article is **Fake** or **Real**.")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    month = st.selectbox("Month", encoder.categories_[0].tolist())
    day = st.slider("Day of Month", 1, 31, 1)

with col2:
    title = st.text_area("News Title", height=80)
    text = st.text_area("News Body Text", height=200)

# Full preprocessing pipeline
def preprocess(title, text, day, month):
    title_len = len(title)
    text_len = len(text)
    word_count_title = len(title.split())
    word_count_text = len(text.split())
    title_reuse_count = text.lower().count(title.lower().split()[0]) if title else 0

    cleaned_text = clean_text(text)
    cleaned_title = clean_text(title)

    # TF-IDF
    X_text = vectorizer.transform([cleaned_text])

    # Encoded categorical
    encoded_month = encoder.transform([[month]])

    # Numeric features
    numeric_df = pd.DataFrame([[day, title_len, text_len, word_count_title, word_count_text, title_reuse_count]],
                               columns=["day", "title_len", "text_len", "word_count_title", "word_count_text", "title_reuse_count"])
    scaled_numeric = scaler.transform(numeric_df)

    # Final input
    X_final = np.hstack([X_text.toarray(), encoded_month, scaled_numeric])
    return X_final, cleaned_text, numeric_df.columns.tolist()

# --- Predict Button ---
if st.button("Analyze News"):
    with st.spinner("Processing and predicting..."):
        time.sleep(1.5)

        X_final, cleaned_text, numeric_cols = preprocess(title, text, day, month)
        pred = model.predict(X_final)[0]
        prob = model.predict_proba(X_final)[0]

        # Show prediction
        st.success(f"Prediction: {'Real News' if pred==1 else 'Fake News'}")
        st.info(f"Confidence: {max(prob)*100:.2f}%")
with st.expander("View Model Evaluation Metrics (Test Set)"):
    st.markdown("#### Classification Report")
    st.markdown("""
        | Class     | Precision | Recall | F1-score | Support |
        |-----------|-----------|--------|----------|---------|
        | **Fake (0)** | 0.97      | 0.97   | 0.97     | 3496    |
        | **Real (1)** | 0.98      | 0.97   | 0.97     | 4160    |
        |             |           |        |          |         |
        | **Accuracy**     |           |        | **0.97**   | 7656    |
        | **Macro Avg**    | 0.97      | 0.97   | 0.97     | 7656    |
        | **Weighted Avg** | 0.97      | 0.97   | 0.97     | 7656    |
        """)


    st.markdown("#### Confusion Matrix Heatmap")

    # Manually entered confusion matrix (based on your results)
    cm = [[3383, 113],
        [127, 4033]]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    st.markdown("#### ROC AUC Score: **0.9940**")


st.markdown("---")
st.markdown("*Model trained with TF-IDF + OneHot + Scaled Features using Optuna-tuned Naive Bayes.*")

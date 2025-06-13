# Fake News Detection + News Source Analysis

A powerful machine learning project that detects **Fake vs Real News** using advanced text preprocessing, feature engineering, and a Multinomial Naive Bayes classifier. Also includes source-based analysis.

---

## Features

- Advanced text preprocessing (cleaning, lemmatization, stopword removal)
- TF-IDF vectorization with bi-grams
- Multinomial Naive Bayes classifier (97% accuracy)
- Feature engineering (text stats, title reuse, one-hot month encoding)
- Scaled numerical features
- Streamlit web app for live predictions

---

## Project Structure

- `app.py` — Streamlit app  
- `best_fake_news_model.pkl` — Trained Naive Bayes model  
- `news_tfidf_vectorizer.pkl` — Fitted TF-IDF vectorizer  
- `news_encoder.pkl` — OneHotEncoder for 'month'  
- `news_scaler.pkl` — Scaler for numeric features  
- `cleaned_news_dataset.csv` — Cleaned and labeled dataset  
- `requirements.txt` — Python dependencies  
- `README.md` — Project documentation

## Model Performance

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| **Fake (0)** | 0.964     | 0.968  | 0.966    | 3496    |
| **Real (1)** | 0.973     | 0.969  | 0.971    | 4160    |
|             |           |        |          |         |
| **Accuracy**     |           |        | **0.9687** | 7656    |
| **Macro Avg**    | 0.968     | 0.969  | 0.968    | 7656    |
| **Weighted Avg** | 0.969     | 0.969  | 0.969    | 7656    |

**ROC AUC Score**: 0.9940  
**Confusion Matrix**

| Actual \ Predicted | Fake | Real |
|--------------------|------|------|
| **Fake**           | 3383 | 113  |
| **Real**           | 127  | 4033 |

---

## Preprocessing & Features

- **TF-IDF** with `max_features=25000`, bi-grams, `sublinear_tf=True`, *no IDF*
- **OneHotEncoder** for `month`
- **Scaled numeric features**:
  - `day`, `title_len`, `text_len`, `word_count_title`, `word_count_text`, `title_reuse_count`
- **Cleaned Text**:
  - Lowercasing  
  - Punctuation & number removal  
  - Stopwords removal  
  - Lemmatization  
  - HTML/URL stripping
---

Developed by [Zeeshan Akram](https://github.com/zeeshan-akram-ds) — aspiring Data Scientist.

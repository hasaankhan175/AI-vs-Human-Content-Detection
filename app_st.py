
import streamlit as st
# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI vs Human Text Detector", page_icon="🤖", layout="centered")

import numpy as np
import pandas as pd
import torch
import re
from wordfreq import word_frequency
from textblob import TextBlob
import textstat
import spacy
from transformers import BertTokenizer, BertModel
import joblib

# ---------------- Load Models ----------------
@st.cache_resource
def load_resources():
    nlp = spacy.load("en_core_web_sm")
    rf_model = joblib.load(r"D:\portfolio_of_data_science\contentdetection\random_forest_ai_human.pkl")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    return nlp, rf_model, tokenizer, bert_model

nlp, rf_model, tokenizer, bert_model = load_resources()

# ---------------- BERT Embedding ----------------
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# ---------------- Feature Computation ----------------
def compute_numeric_features(text):
    doc = nlp(text)
    
    grammar_errors = 0
    passive_sent_count = sum(1 for sent in doc.sents if any(tok.dep_ == "nsubjpass" for tok in sent))
    passive_voice_ratio = passive_sent_count / max(len(list(doc.sents)), 1)
    
    words_clean = [token.text.lower() for token in doc if token.is_alpha]
    predictability_score = np.mean([word_frequency(w, 'en') for w in words_clean]) if words_clean else 0
    
    sentence_lengths = [len([t for t in sent if t.is_alpha]) for sent in doc.sents]
    burstiness = np.var(sentence_lengths) if sentence_lengths else 0
    
    sentiment_score = TextBlob(text).sentiment.polarity
    
    flesch = textstat.flesch_reading_ease(text)
    fog = textstat.gunning_fog(text)
    
    return np.array([
        len(words_clean),                            # word_count
        len(text),                                   # character_count
        len(list(doc.sents)),                        # sentence_count
        len(set(words_clean)) / len(words_clean),    # lexical_diversity
        len(words_clean) / max(len(list(doc.sents)), 1),  # avg_sentence_length
        np.mean([len(w) for w in words_clean]),      # avg_word_length
        sum(1 for c in text if c in '.,;!?') / len(text),  # punctuation_ratio
        flesch,                                      # flesch_reading_ease
        fog,                                         # gunning_fog_index
        grammar_errors,                              # grammar_errors
        passive_voice_ratio,                         # passive_voice_ratio
        predictability_score,                        # predictability_score
        burstiness,                                  # burstiness
        sentiment_score                              # sentiment_score
    ]), {
        "Word Count": len(words_clean),
        "Character Count": len(text),
        "Sentence Count": len(list(doc.sents)),
        "Lexical Diversity": round(len(set(words_clean)) / len(words_clean), 3) if words_clean else 0,
        "Avg Sentence Length": round(len(words_clean) / max(len(list(doc.sents)), 1), 3),
        "Avg Word Length": round(np.mean([len(w) for w in words_clean]), 3) if words_clean else 0,
        "Punctuation Ratio": round(sum(1 for c in text if c in '.,;!?') / len(text), 3) if text else 0,
        "Flesch Reading Ease": flesch,
        "Gunning Fog Index": fog,
        "Passive Voice Ratio": round(passive_voice_ratio, 3),
        "Predictability Score": round(predictability_score, 6),
        "Burstiness": round(burstiness, 3),
        "Sentiment Score": round(sentiment_score, 3)
    }


st.title("🤖 AI vs Human Text Detector")
st.write("Paste your text below to check if it’s **AI-written** or **Human-written**.")

user_text = st.text_area("✏️ Enter your text here:", height=200)

if st.button("Analyze Text"):
    if user_text.strip():
        # Get features
        numeric_features, feature_dict = compute_numeric_features(user_text)
        bert_emb = get_bert_embedding(user_text)
        all_features = np.hstack([numeric_features.reshape(1, -1), bert_emb])
        
        # Prediction
        pred_label = rf_model.predict(all_features)[0]
        pred_text = "🟢 Likely AI-written" if pred_label == 1 else "🔵 Likely Human-written"
        
        st.subheader("📌 Prediction Result")
        st.markdown(f"### {pred_text}")
        
        # Show numeric features
        st.subheader("📊 Extracted Numeric Features")
        st.table(pd.DataFrame(list(feature_dict.items()), columns=["Feature", "Value"]))
    else:
        st.warning("Please enter some text to analyze.")

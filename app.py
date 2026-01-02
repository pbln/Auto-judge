import streamlit as st
import numpy as np
import pandas as pd
import re
import joblib
from scipy.sparse import hstack, csr_matrix

@st.cache_resource
def load_models():
    tfidf = joblib.load("model/tfidf.pkl")
    clf = joblib.load("model/classifier.pkl")
    reg = joblib.load("model/regressor.pkl")
    return tfidf, clf, reg

tfidf, clf, reg = load_models()

def clean_text(text: str) -> str:
    text = text.replace("$", "")
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_structural_features(text: str):
    words = text.split()

    log_num_words = np.log1p(len(words))
    num_sentences = np.log1p(text.count("."))

    return np.array([log_num_words, num_sentences])


SPATIAL_WORDS = [
    "grid", "matrix", "cell", "square",
    "row", "column",
    "coordinate", "position", "location",
    "distance", "path", "move", "step",
    "neighbor", "adjacent", "direction",
    "north", "south", "left", "right",
    "x", "y"
]

def spatial_count(text: str) -> int:
    text = text.lower()
    return sum(text.count(w) for w in SPATIAL_WORDS)


def fet2(text: str) -> float:
    nums = [int(n) for n in re.findall(r"\d+", text)]
    return np.log1p(max(nums)) if nums else 0.0


CLASS_RANGES = {
    "easy": (1.1 , 2.8),
    "medium": (2.8 , 5.5),
    "hard": (5.5 , 9.7),
}

def denormalize_score(y_rel, cls):
    lo, hi = CLASS_RANGES[cls]
    y_rel = np.clip(y_rel, 0, 1)
    return lo + y_rel * (hi - lo)

st.set_page_config(page_title="Problem Difficulty Analyzer", layout="centered")

st.title("📘 Problem Difficulty Analyzer")
st.markdown("Predict **problem class** and **difficulty score**.")

desc = st.text_area("Problem Description", height=150)
inp = st.text_area("Input Description", height=100)
out = st.text_area("Output Description", height=100)

if st.button("🔍 Analyze"):
    if not desc.strip():
        st.warning("Please enter problem description")
    else:
        
        full_text = f"{desc} {inp} {out}"
        full_text2 = clean_text(full_text)

        X_text = tfidf.transform([full_text2])

       
        struct = extract_structural_features(full_text)         
        spatial = spatial_count(full_text)                        
        maxnum = fet2(full_text)                               

        X_num = np.array([[struct[0], struct[1], maxnum]])

        X_final = hstack([
            X_text,
            X_num,
            csr_matrix(np.array(spatial))
        ])

    
        pred_class = clf.predict(X_final)[0]
        pred_score = reg.predict(X_final)[0]


        st.success("Prediction Complete")
        st.metric("Predicted Class", pred_class)
        st.metric("Predicted Score", round(denormalize_score(pred_score , pred_class) , 3))

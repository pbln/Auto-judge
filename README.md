# Problem Difficulty Predictor

This project predicts the **difficulty of programming problems** using only textual information from problem statements.  
It performs two tasks:

1. **Classification** – Predicts the difficulty class  
   (`Easy`, `Medium`, `Hard`)
2. **Regression** – Predicts a continuous difficulty score

The system is trained using classical machine learning models and is deployed through a simple web interface.

---

## Project Overview

Programming problem difficulty is often subjective and depends on problem structure, constraints, and reasoning complexity.  
This project approximates difficulty by extracting **textual and structural cues** from problem descriptions, without using any code submissions or metadata.

The dataset is relatively small (~4000 samples), so the focus is on:
- Robust feature engineering
- Simple, interpretable models
- Careful train–inference consistency

---

## Features Used

All features are derived **only from text**.

### Textual Features
- **TF-IDF vectors** (unigrams and bigrams)
- Lowercased text with basic symbol cleaning

### Structural / Numeric Features
- `log(number of words)`
- `number of sentences`
- **Spatial keyword count**  
  (e.g. grid, distance, row, column, matrix)
- **Numeric constraint magnitude**  
  (log of maximum number appearing in the text)

These features were chosen after extensive experimentation and ablation studies.

---

## Models Used

### Classification
- **Logistic Regression**
  - Chosen for its stability and strong performance on sparse TF-IDF features

### Regression
- **Ridge Regression**
  - Predicts a relative difficulty score
  - Final score is calibrated using the predicted class to ensure consistency

This two-stage approach produces stable and interpretable predictions.

---

## Evaluation

### Classification
- Accuracy
- Confusion matrix

### Regression
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Due to the subjective nature of difficulty scores, absolute regression accuracy is limited.  
Class-aware calibration is used to improve score reliability.

---

## Web Interface

A simple **Streamlit** interface is provided where users can enter:
- Problem description
- Input description
- Output description

The app displays:
- Predicted difficulty class
- Predicted difficulty score

---


---

## How to Run Locally

```bash
git clone https://github.com/your-username/Auto-judge.git
cd Auto-judge

2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

3. Install dependencies
pip install -r req.txt

4. Run the web app
streamlit run app.py

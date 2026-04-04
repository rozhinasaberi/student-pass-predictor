#  Student Pass Predictor

A machine learning web application that predicts whether a student will **pass or fail** based on academic, social, and lifestyle factors.

 **Live App:**  
[(https://student-pass-predictor-rojinasaberiapps.streamlit.app/)](https://student-pass-predictor-rojinasaberiapps.streamlit.app/)

---

##  Project Overview

This project uses a **Decision Tree Classifier** built with Scikit-learn to analyze student performance data and predict outcomes.

The dataset comes from the **UCI Machine Learning Repository – Student Performance Dataset**.

---

##  Features

- Predict pass/fail based on student inputs
- Clean and user-friendly web interface
- Real-time predictions
- Uses a trained machine learning pipeline
- Deployed using Streamlit Cloud

---

##  Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

---

##  Dataset

- Source: UCI Student Performance Dataset
- File used: `student-por.csv`
- Total records: 649 students

---

##  Model Details

- Algorithm: Decision Tree Classifier
- Criterion: Entropy
- Tuned using RandomizedSearchCV
- Final pipeline includes:
  - One-hot encoding for categorical variables
  - Decision tree model

---

##  Performance

| Metric      | Score |
|------------|------|
| Accuracy   | ~69% |
| Precision  | ~71% |
| Recall     | ~69% |

---

##  Key Insight

The most important feature in predicting student success was:

 **Number of past failures**

Students with previous failures were significantly more likely to fail again.

---

##  Files Included

- `app.py` → Streamlit web app
- `pipeline_rojina.pkl` → full ML pipeline
- `best_model_rojina.pkl` → trained model
- `requirements.txt` → dependencies
- `student-por.csv` → dataset
- `decision_tree_rojina.png` → model visualization

---

##  How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

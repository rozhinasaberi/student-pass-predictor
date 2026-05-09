# Student Pass Predictor

A machine learning Streamlit application that predicts whether a student is likely to **pass or fail** based on academic, social, and lifestyle factors.

Live demo: [student-pass-predictor-rojinasaberiapps.streamlit.app](https://student-pass-predictor-rojinasaberiapps.streamlit.app/)

## What This Project Is About

This project uses supervised learning to analyze student performance patterns and generate a pass/fail prediction from user-provided inputs. It is designed as an applied machine learning project with a simple web interface for real-time predictions.

This is an **AI / machine learning classification project** and a **deployed Streamlit web app**.

## Model Overview

- Algorithm: Decision Tree Classifier
- Criterion: Entropy
- Tuning method: RandomizedSearchCV
- Pipeline: one-hot encoding plus trained decision tree model

## Dataset

- Source: UCI Student Performance Dataset
- File used: `student-por.csv`
- Records: 649 students

## Performance

- Accuracy: about 69%
- Precision: about 71%
- Recall: about 69%

One of the main findings from the project is that **past failures** were the strongest predictor of student outcome.

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- joblib

## Main Files

- `app.py` - Streamlit app
- `pipeline_rojina.pkl` - full ML pipeline
- `best_model_rojina.pkl` - trained model
- `student-por.csv` - dataset
- `decision_tree_rojina.png` - tree visualization
- `requirements.txt` - dependencies

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

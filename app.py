from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .block-container {
            max-width: 1050px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: rgba(255,255,255,0.82);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            color: #0f172a;
            font-size: 2rem;
            line-height: 1.2;
        }
        .hero p {
            margin: 0.45rem 0 0 0;
            color: #475569;
            font-size: 1rem;
        }
        .card {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.2rem 1.2rem 0.6rem 1.2rem;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }
        .section-title {
            color: #0f172a;
            font-size: 1.08rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        .stSelectbox label, .stSlider label {
            color: #0f172a !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }
        .stMarkdown, .stCaption, p, div {
            color: #0f172a;
        }
        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #dbe3f0 !important;
        }
        div[data-baseweb="select"] * {
            color: #0f172a !important;
        }
        div[role="combobox"] {
            background: #ffffff !important;
            color: #0f172a !important;
        }
        div[data-baseweb="popover"] * {
            background: #ffffff !important;
            color: #0f172a !important;
        }
        div[role="listbox"] * {
            background: #ffffff !important;
            color: #0f172a !important;
        }
        .result-pass {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.28);
            color: #166534;
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 0.4rem;
        }
        .result-fail {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.28);
            color: #991b1b;
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 0.4rem;
        }
        .metric-line {
            color: #475569;
            font-size: 0.95rem;
            margin-top: 0.35rem;
        }
        .footer-note {
            text-align: center;
            color: #64748b;
            font-size: 0.92rem;
            margin-top: 0.5rem;
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            border-radius: 16px;
            height: 3.1rem;
            background: #4f46e5;
            color: white;
            font-weight: 700;
            border: none;
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.25);
        }
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #4338ca;
            color: white;
        }
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(135deg, #111827 0%, #334155 100%);
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURE_ORDER = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]

SCHOOL_MAP = {"Gabriel Pereira": "GP", "Mousinho da Silveira": "MS"}
SEX_MAP = {"Female": "F", "Male": "M"}
ADDRESS_MAP = {"Urban": "U", "Rural": "R"}
FAMSIZE_MAP = {"More than 3": "GT3", "3 or fewer": "LE3"}
PSTATUS_MAP = {"Together": "T", "Apart": "A"}
JOB_MAP = {
    "At home": "at_home",
    "Health": "health",
    "Other": "other",
    "Services": "services",
    "Teacher": "teacher",
}
REASON_MAP = {
    "Course preference": "course",
    "Close to home": "home",
    "Other": "other",
    "School reputation": "reputation",
}
GUARDIAN_MAP = {"Mother": "mother", "Father": "father", "Other": "other"}
YES_NO_MAP = {"No": "no", "Yes": "yes"}
EDUCATION_MAP = {
    "No formal education": 0,
    "Primary school": 1,
    "Middle school": 2,
    "High school": 3,
    "College or university": 4,
}

@st.cache_resource
def load_model():
   return joblib.load(PROJECT_DIR / "pipeline_rojina.pkl")


def build_input_row(values: dict) -> pd.DataFrame:
    row = {feature: values[feature] for feature in FEATURE_ORDER}
    return pd.DataFrame([row])


def main():
    st.markdown(
        """
        <div class="hero">
            <h1>🎓 Student Pass Predictor</h1>
            <p>Enter student details below to get a simple pass or fail prediction from your tuned decision tree model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Basic Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            school_display = st.selectbox("School", list(SCHOOL_MAP.keys()), index=0)
            sex_display = st.selectbox("Sex", list(SEX_MAP.keys()), index=0)
            age = st.slider("Age", 15, 22, 17)
        with c2:
            address_display = st.selectbox("Address", list(ADDRESS_MAP.keys()), index=0)
            famsize_display = st.selectbox("Family size", list(FAMSIZE_MAP.keys()), index=0)
            pstatus_display = st.selectbox("Parents' status", list(PSTATUS_MAP.keys()), index=0)
        with c3:
            guardian_display = st.selectbox("Guardian", list(GUARDIAN_MAP.keys()), index=0)
            higher_display = st.selectbox("Wants higher education", list(YES_NO_MAP.keys()), index=1)
            internet_display = st.selectbox("Internet access", list(YES_NO_MAP.keys()), index=1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Family and School Background</div>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            medu_display = st.selectbox("Mother's education", list(EDUCATION_MAP.keys()), index=2)
            fedu_display = st.selectbox("Father's education", list(EDUCATION_MAP.keys()), index=2)
            reason_display = st.selectbox("Reason for choosing school", list(REASON_MAP.keys()), index=0)
        with c5:
            mjob_display = st.selectbox("Mother's job", list(JOB_MAP.keys()), index=2)
            fjob_display = st.selectbox("Father's job", list(JOB_MAP.keys()), index=2)
            traveltime = st.slider("Travel time", 1, 4, 2)
        with c6:
            studytime = st.slider("Study time", 1, 4, 2)
            schoolsup_display = st.selectbox("School support", list(YES_NO_MAP.keys()), index=0)
            famsup_display = st.selectbox("Family support", list(YES_NO_MAP.keys()), index=1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Lifestyle and Performance Factors</div>', unsafe_allow_html=True)
        c7, c8, c9 = st.columns(3)
        with c7:
            failures = st.slider("Past class failures", 0, 4, 0)
            absences = st.slider("Absences", 0, 40, 4)
            famrel = st.slider("Family relationship quality", 1, 5, 4)
        with c8:
            freetime = st.slider("Free time", 1, 5, 3)
            goout = st.slider("Going out with friends", 1, 5, 3)
            health = st.slider("Health status", 1, 5, 4)
        with c9:
            dalc = st.slider("Workday alcohol use", 1, 5, 1)
            walc = st.slider("Weekend alcohol use", 1, 5, 2)
            romantic_display = st.selectbox("In a relationship", list(YES_NO_MAP.keys()), index=0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Additional Support</div>', unsafe_allow_html=True)
        c10, c11, c12 = st.columns(3)
        with c10:
            paid_display = st.selectbox("Extra paid classes", list(YES_NO_MAP.keys()), index=0)
        with c11:
            activities_display = st.selectbox("Extra activities", list(YES_NO_MAP.keys()), index=1)
        with c12:
            nursery_display = st.selectbox("Attended nursery", list(YES_NO_MAP.keys()), index=1)
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Predict")

    values = {
        "school": SCHOOL_MAP[school_display],
        "sex": SEX_MAP[sex_display],
        "age": age,
        "address": ADDRESS_MAP[address_display],
        "famsize": FAMSIZE_MAP[famsize_display],
        "Pstatus": PSTATUS_MAP[pstatus_display],
        "Medu": EDUCATION_MAP[medu_display],
        "Fedu": EDUCATION_MAP[fedu_display],
        "Mjob": JOB_MAP[mjob_display],
        "Fjob": JOB_MAP[fjob_display],
        "reason": REASON_MAP[reason_display],
        "guardian": GUARDIAN_MAP[guardian_display],
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": YES_NO_MAP[schoolsup_display],
        "famsup": YES_NO_MAP[famsup_display],
        "paid": YES_NO_MAP[paid_display],
        "activities": YES_NO_MAP[activities_display],
        "nursery": YES_NO_MAP[nursery_display],
        "higher": YES_NO_MAP[higher_display],
        "internet": YES_NO_MAP[internet_display],
        "romantic": YES_NO_MAP[romantic_display],
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": dalc,
        "Walc": walc,
        "health": health,
        "absences": absences,
    }

    if submitted:
        try:
            model = load_model()
            input_df = build_input_row(values)
            prediction = model.predict(input_df)[0]

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

            if prediction == 1:
                st.markdown("<div class='result-pass'>Prediction: Pass</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-fail'>Prediction: Fail</div>", unsafe_allow_html=True)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                fail_prob = float(probs[0])
                pass_prob = float(probs[1])
                st.markdown(
                    f"<div class='metric-line'>Pass probability: {pass_prob:.2%} &nbsp; | &nbsp; Fail probability: {fail_prob:.2%}</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("See submitted data"):
                preview_df = pd.DataFrame([{
                    "School": school_display,
                    "Sex": sex_display,
                    "Age": age,
                    "Address": address_display,
                    "Family size": famsize_display,
                    "Parents' status": pstatus_display,
                    "Mother's education": medu_display,
                    "Father's education": fedu_display,
                    "Mother's job": mjob_display,
                    "Father's job": fjob_display,
                    "Reason": reason_display,
                    "Guardian": guardian_display,
                    "Travel time": traveltime,
                    "Study time": studytime,
                    "Past failures": failures,
                    "School support": schoolsup_display,
                    "Family support": famsup_display,
                    "Paid classes": paid_display,
                    "Activities": activities_display,
                    "Nursery": nursery_display,
                    "Higher education": higher_display,
                    "Internet": internet_display,
                    "Romantic": romantic_display,
                    "Family relationship": famrel,
                    "Free time": freetime,
                    "Going out": goout,
                    "Workday alcohol": dalc,
                    "Weekend alcohol": walc,
                    "Health": health,
                    "Absences": absences,
                }])
                st.dataframe(preview_df, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        except FileNotFoundError:
            st.error("Could not find best_model_rojina.pkl. Put the model file in the same folder as this app.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    st.markdown("<div class='footer-note'>Built with Streamlit and your tuned decision tree model.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
PROJECT_DIR = Path(__file__).resolve().parent

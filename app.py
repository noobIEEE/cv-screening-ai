import streamlit as st
import pdfplumber
import zipfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# PDF READER
# =========================
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# =========================
# SKILL DICTIONARY (simple but effective)
# =========================
SKILL_DB = [
    "python", "sql", "machine learning", "data analysis",
    "excel", "tableau", "power bi", "deep learning",
    "communication", "management", "java", "javascript"
]

def extract_skills(text):
    text = text.lower()
    found = [skill for skill in SKILL_DB if skill in text]
    return found

# =========================
# REQUIREMENT SCORE
# =========================
def req_score(text, req):
    if not req.strip():
        return 0
    words = req.lower().split()
    return sum(1 for w in words if w in text.lower()) / len(words)

# =========================
# EXPLAINABLE AI (RULE BASED)
# =========================
def explain(text, req):
    text_low = text.lower()
    req_words = req.lower().split()

    matched = [w for w in req_words if w in text_low]
    missing = [w for w in req_words if w not in text_low]

    explanation = ""
    if matched:
        explanation += "Matched: " + ", ".join(matched) + ". "
    if missing:
        explanation += "Missing: " + ", ".join(missing)

    return explanation if explanation else "No requirement provided"

# =========================
# SESSION
# =========================
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.vectorizer = None

st.title("📄 AI CV Screening Pro (ATS+)")

# =========================
# TRAIN
# =========================
st.header("1️- Teach AI")

ok_files = st.file_uploader("Upload CV OK", accept_multiple_files=True)
ng_files = st.file_uploader("Upload CV NOT OK", accept_multiple_files=True)

if st.button("Train AI"):
    texts, labels = [], []

    for f in ok_files:
        texts.append(extract_text(f))
        labels.append(1)

    for f in ng_files:
        texts.append(extract_text(f))
        labels.append(0)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    st.session_state.model = model
    st.session_state.vectorizer = vectorizer

    st.success("AI trained!")

# =========================
# SETTINGS
# =========================
st.header("2️- Settings")

requirement = st.text_area("Job Requirement (optional)")

weight_ml = st.slider("ML Weight %", 0, 100, 60)
weight_req = st.slider("Requirement Weight %", 0, 100 - weight_ml, 30)
weight_skill = 100 - (weight_ml + weight_req)

top_n = st.slider("Top N Candidates", 1, 20, 10)

# =========================
# SCREENING
# =========================
st.header("3- Screening CV")

files = st.file_uploader("Upload CVs", accept_multiple_files=True)

if st.button("Run Screening"):

    results = []

    for f in files:
        text = extract_text(f)

        X = st.session_state.vectorizer.transform([text])
        ml = st.session_state.model.predict_proba(X)[0][1]

        req = req_score(text, requirement)
        skills = extract_skills(text)

        skill_score = len(skills) / len(SKILL_DB)

        final = (
            (ml * weight_ml / 100) +
            (req * weight_req / 100) +
            (skill_score * weight_skill / 100)
        )

        explanation = explain(text, requirement)

        results.append({
            "name": f.name,
            "ml_score": ml,
            "req_score": req,
            "skill_score": skill_score,
            "skills_found": ", ".join(skills),
            "final_score": final,
            "explanation": explanation
        })

    # =========================
    # SORT TOP N
    # =========================
    df = pd.DataFrame(results)
    df = df.sort_values("final_score", ascending=False)

    st.subheader("🏆 Top Candidates")
    st.dataframe(df.head(top_n))

    st.subheader("📊 Full Results")
    st.dataframe(df)

    # =========================
    # EXPORT
    # =========================
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False),
        "results.csv"
    )

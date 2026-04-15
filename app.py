import streamlit as st
import pdfplumber
import zipfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# 📄 BACA PDF
# =========================
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# =========================
# 🧠 SIMPAN MODEL
# =========================
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.vectorizer = None

st.title("📄 CV Screening AI (Simple ATS)")

# =========================
# 🧠 TEACH AI
# =========================
st.header("1️⃣ Teaching AI")

ok_files = st.file_uploader("Upload CV OK", accept_multiple_files=True)
ng_files = st.file_uploader("Upload CV NOT OK", accept_multiple_files=True)

if st.button("Train AI"):
    texts = []
    labels = []

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

    st.success("AI sudah belajar!")

# =========================
# 📋 REQUIREMENT
# =========================
st.header("2️⃣ Requirement Pekerjaan")

requirement = st.text_area("Contoh: python data analyst sql 3 years")

def req_score(text, req):
    words = req.lower().split()
    score = sum(1 for w in words if w in text.lower())
    return score / len(words) if words else 0

# =========================
# 🔍 SCREENING
# =========================
st.header("3️⃣ Screening CV")

screen_files = st.file_uploader("Upload CV untuk screening", accept_multiple_files=True)

if st.button("Run Screening"):
    if st.session_state.model is None:
        st.error("Train AI dulu!")
    else:
        results = []
        zip_ok = zipfile.ZipFile("OK.zip", "w")
        zip_ng = zipfile.ZipFile("NG.zip", "w")

        for f in screen_files:
            text = extract_text(f)
            X = st.session_state.vectorizer.transform([text])

            prob = st.session_state.model.predict_proba(X)[0][1]
            r_score = req_score(text, requirement)

            final = (prob * 0.7) + (r_score * 0.3)

            status = "OK" if final > 0.5 else "NOT OK"

            results.append({
                "Nama": f.name,
                "AI Score": round(prob,2),
                "Req Score": round(r_score,2),
                "Final Score": round(final,2),
                "Status": status
            })

            if status == "OK":
                zip_ok.writestr(f.name, f.read())
            else:
                zip_ng.writestr(f.name, f.read())

        zip_ok.close()
        zip_ng.close()

        df = pd.DataFrame(results)
        df.to_csv("hasil.csv", index=False)

        st.dataframe(df)

        st.download_button("⬇️ Download CSV", open("hasil.csv","rb"), "hasil.csv")
        st.download_button("⬇️ Download OK CV", open("OK.zip","rb"), "OK.zip")
        st.download_button("⬇️ Download NG CV", open("NG.zip","rb"), "NG.zip")

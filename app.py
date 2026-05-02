import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------- TITLE --------
st.title("AI Resume Analyzer")

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:

    # -------- READ PDF --------
    pdf = PyPDF2.PdfReader(uploaded_file)
    resume_text = ""

    for page in pdf.pages:
        if page.extract_text():
            resume_text += page.extract_text()

    resume_text = resume_text.encode('utf-8', errors='ignore').decode('utf-8')
    resume_text = resume_text.lower()

    # -------- SKILL EXTRACTION --------
    skills_list = [
        "python", "machine learning", "data analysis",
        "excel", "power bi", "html", "css",
        "java", "sql", "nlp"
    ]

    found_skills = [skill for skill in skills_list if skill in resume_text]

    # -------- MODEL --------
    data = [
        ("python machine learning data analysis", "Data Scientist"),
        ("python data analysis excel power bi", "Data Analyst"),
        ("html css javascript", "Frontend Developer"),
        ("java sql backend", "Backend Developer")
    ]

    texts, labels = zip(*data)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    # -------- PREDICTION --------
    input_text = " ".join(found_skills)
    prediction = model.predict(vectorizer.transform([input_text]))
    role = prediction[0]

    # -------- ANALYSIS --------
    job_roles = {
        "Data Scientist": ["python", "machine learning", "data analysis", "nlp"],
        "Data Analyst": ["python", "data analysis", "excel", "power bi"],
        "Frontend Developer": ["html", "css", "javascript"],
        "Backend Developer": ["java", "sql"]
    }

    required = job_roles.get(role, [])

    matched = [s for s in found_skills if s in required]
    missing = [s for s in required if s not in found_skills]

    score = (len(matched) / len(required)) * 100 if required else 0

    # -------- OUTPUT --------
    st.subheader("Analysis Report")

    st.write("**Extracted Skills:**", found_skills)
    st.write("**Predicted Role:**", role)
    st.write("**Resume Score:**", round(score, 2), "%")
    st.write("**Missing Skills:**", missing)

import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from xhtml2pdf import pisa
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

# -------- SIDEBAR --------
st.sidebar.title("📌 About Project")
st.sidebar.write("AI Resume Analyzer\n and job role perdictor")

# -------- TITLE --------
st.markdown("<h1 style='text-align:center;color:#4CAF50;'>AI Resume Analyzer 🚀</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:

    # -------- READ PDF --------
    pdf = PyPDF2.PdfReader(uploaded_file)
    text = "".join([p.extract_text() or "" for p in pdf.pages]).lower()
    st.success("✅ Resume Uploaded")

    # -------- SKILLS --------
    skills_list = ["python","machine learning","data analysis","excel","power bi","html","css","java","sql","nlp"]
    skills = [s for s in skills_list if s in text]

    # -------- MODEL --------
    data = [
        ("python machine learning data analysis","Data Scientist"),
        ("python data analysis excel power bi","Data Analyst"),
        ("html css javascript","Frontend Developer"),
        ("java sql backend","Backend Developer")
    ]
    X, y = zip(*data)
    vec = CountVectorizer()
    model = MultinomialNB().fit(vec.fit_transform(X), y)
    role = model.predict(vec.transform([" ".join(skills)]))[0]

    # -------- UI (ROLE + SCORE) --------
    job_roles = {
        "Data Scientist": ["python","machine learning","data analysis","nlp"],
        "Data Analyst": ["python","data analysis","excel","power bi"],
        "Frontend Developer": ["html","css","javascript"],
        "Backend Developer": ["java","sql"]
    }

    req = job_roles.get(role, [])
    matched = [s for s in skills if s in req]
    missing = [s for s in req if s not in skills]
    score = (len(matched)/len(req))*100 if req else 0

    col1, col2 = st.columns(2)
    col1.success(f"💼 {role}")
    col2.metric("📈 Score", f"{round(score,2)} %")

    # -------- SKILL CATEGORY --------
    st.subheader("🧠 Skill Categories")
    st.write("💻 Tech:", [s for s in skills if s in ["python","machine learning","nlp","sql"]])
    st.write("🌐 Web:", [s for s in skills if s in ["html","css"]])
    st.write("🛠 Tools:", [s for s in skills if s in ["excel","power bi"]])

    # -------- EXPERIENCE --------
    level = "Fresher / Intern" if "intern" in text else "Mid Level" if "5 years" in text else "Entry Level"
    st.subheader("📊 Experience Level")
    st.success(level)

    # -------- MISSING --------
    st.subheader("⚠️ Missing Skills")
    if missing:
        [st.error(s) for s in missing]
    else:
        st.success("No missing skills 🎉")

    # -------- SUGGESTIONS --------
    st.subheader("💡 Suggestions")
    if missing:
        [st.warning(f"Add {s}") for s in missing]
    else:
        st.success("Excellent Resume 🚀")

    # -------- CHART --------
    st.subheader("📊 Skills Chart")
    if skills:
        fig, ax = plt.subplots()
        ax.bar(skills, [1]*len(skills))
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # -------- PDF --------
     # -------- PDF FUNCTION --------
def make_pdf():
    missing_text = ", ".join(missing) if missing else "None"
    skills_text = ", ".join(skills) if skills else "None"

    html = f"""
    <html>
    <body>
    <h2>AI Resume Analyzer</h2>
    <p><b>Role:</b> {role}</p>
    <p><b>Level:</b> {level}</p>
    <p><b>Score:</b> {round(score,2)}%</p>
    <p><b>Skills:</b> {skills_text}</p>
    <p><b>Missing:</b> {missing_text}</p>
    </body>
    </html>
    """

    buffer = io.BytesIO()
    pisa.CreatePDF(html, dest=buffer)
    buffer.seek(0)
    return buffer

    st.subheader("📄 Download Report")
    st.download_button("⬇ Download PDF", make_pdf(), "report.pdf", "application/pdf")

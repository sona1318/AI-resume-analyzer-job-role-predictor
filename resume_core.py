import PyPDF2

# Extract text from PDF resume
def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    return text


# Simple job prediction logic
def predict_role(text):
    text = text.lower()

    if "python" in text and "machine learning" in text:
        return "Data Scientist"
    elif "html" in text and "css" in text:
        return "Web Developer"
    elif "android" in text or "kotlin" in text:
        return "Android Developer"
    elif "excel" in text or "finance" in text:
        return "Data Analyst"
    else:
        return "Fresher / IT General Role"

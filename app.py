from flask import Flask, request, render_template
import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords

app = Flask(__name__)
UPLOAD_FOLDER = 'resumes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text.lower()

def score_resume(resume_text, jd_text):
    documents = [resume_text, jd_text]
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(documents)
    score = (matrix * matrix.T).toarray()[0, 1]
    return round(score * 100, 2)

def get_keywords(text, max_features=20):
    vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    vectorizer.fit([text])
    return set(vectorizer.get_feature_names_out())

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume_file = request.files['resume']
        jd_text = request.form['jd']

        resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(resume_path)

        resume_text = extract_text_from_pdf(resume_path)
        score = score_resume(resume_text, jd_text)

        resume_keywords = get_keywords(resume_text)
        jd_keywords = get_keywords(jd_text)
        missing_keywords = sorted(jd_keywords - resume_keywords)

        feedback = "Excellent match! ✅" if score > 75 else "Good match! 👍" if score > 60 else "Needs improvement 🔍"

        return render_template('index.html',
                               score=score,
                               feedback=feedback,
                               missing=missing_keywords,
                               jd_keywords=sorted(jd_keywords),
                               resume_keywords=sorted(resume_keywords))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

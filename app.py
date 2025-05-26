from flask import Flask, request, render_template, redirect, url_for
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import PyPDF2
import logging
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)

# Set up logging to capture error details
logging.basicConfig(level=logging.DEBUG)

# Initialize NLP pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Global variable to store uploaded file content
uploaded_file_content = ""

def extract_content_from_file(file):
    """
    Extract text content from PDF, TXT, or CSV files.
    """
    try:
        if file.filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(file)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + " "
            return content.strip()
        elif file.filename.endswith('.txt'):
            return file.read().decode('utf-8').strip()
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            return df.to_string(index=False)
        return ""
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return ""

def lda_topic_modeling(content, num_topics=3):
    # Preprocessing
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    content = content.lower()
    content = ' '.join([stemmer.stem(word) for word in content.split() if word not in stop_words])

    # Feature Engineering
    vectorizer = TfidfVectorizer(max_features=1000)
    dtm = vectorizer.fit_transform([content])

    # LDA Model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    
    # Get topics with descriptions
    topics = {}
    for idx, topic_weights in enumerate(lda.components_):
        top_word_indices = topic_weights.argsort()[-5:][::-1]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_word_indices]
        topics[f"Topic {idx + 1}"] = {
            "keywords": top_words,
            "description": f"Topic {idx + 1}: " + ", ".join(top_words)
        }
    return topics

def bert_ner(content):
    """
    Perform Named Entity Recognition with contextual results.
    """
    try:
        entities = ner_pipeline(content)
        organized_entities = {}
        for entity in entities:
            label = entity['entity_group']
            text = entity['word']
            sentence = re.search(rf"([^.]*?{re.escape(text)}[^.]*\.)", content, re.IGNORECASE)
            context = sentence.group() if sentence else "No context available"
            
            if label not in organized_entities:
                organized_entities[label] = []
            organized_entities[label].append({"text": text, "context": context})
        
        return organized_entities
    except Exception as e:
        logging.error(f"Error in BERT NER: {e}")
        return {}

def extract_basic_info_from_resume(text):
    """
    Extract basic information (name, email, phone) from resume text using regex.
    """
    try:
        name = re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)
        email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA0-9-]+\.[a-zA-Z0-9-.]+", text)
        phone = re.search(r"\+?\d[\d -]{8,12}\d", text)
        
        name = name.group() if name else "Not Found"
        email = email.group() if email else "Not Found"
        phone = phone.group() if phone else "Not Found"
        
        return name, email, phone
    except Exception as e:
        logging.error(f"Error extracting basic info: {e}")
        return "Not Found", "Not Found", "Not Found"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Upload a file to process.
    """
    global uploaded_file_content
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded. Please upload a valid file."

        file = request.files['file']
        uploaded_file_content = extract_content_from_file(file)
        if not uploaded_file_content:
            return "Uploaded file is empty or unreadable. Please upload a valid file."

        return redirect(url_for('choose_action'))

    return render_template('upload.html')

@app.route('/choose_action', methods=['GET'])
def choose_action():
    """
    Display the action options to the user.
    """
    return render_template('choose_action.html')

@app.route('/analyze/topic_modeling', methods=['GET'])
def analyze_topic_modeling():
    """
    Perform topic modeling with a specified number of topics.
    """
    global uploaded_file_content
    num_topics = request.args.get('num_topics', 3, type=int)
    result = lda_topic_modeling(uploaded_file_content, num_topics=num_topics)
    return render_template('result.html', action='topic_modeling', result=result)

@app.route('/analyze/<action>', methods=['GET'])
def analyze(action):
    """
    Perform the selected analysis (summary, sentiment analysis, entity recognition, resume info).
    """
    global uploaded_file_content

    result = None
    if action == 'summary':
        try:
            summary = summarizer(uploaded_file_content, max_length=130, min_length=30, do_sample=False)
            result = summary[0]['summary_text'] if summary else "No summary available."
        except Exception as e:
            logging.error(f"Error in summarization: {e}")
            result = "Error generating summary."
    elif action == 'sentiment_analysis':
        try:
            sentiment = sentiment_analyzer(uploaded_file_content)
            result = sentiment
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {e}")
            result = "Error generating sentiment analysis."
    elif action == 'entity_recognition':
        result = bert_ner(uploaded_file_content)
    elif action == 'resume_info':
        try:
            name, email, phone = extract_basic_info_from_resume(uploaded_file_content)
            result = {"name": name, "email": email, "phone": phone}
        except Exception as e:
            logging.error(f"Error extracting resume info: {e}")
            result = "Error extracting resume info."
    else:
        result = "Invalid action selected."

    return render_template('result.html', action=action, result=result)

if __name__ == '__main__':
    app.run(debug=True)

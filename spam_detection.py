# spam_detection.py - FINAL WORKING VERSION
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
import string
import streamlit as st

# Setup NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Define preprocessing function FIRST
def preprocess(text):
    text = str(text).lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Load and train model
@st.cache_resource
def load_model():
    df = pd.read_csv('spam.csv', encoding='latin-1')[['label', 'text']]
    df['clean_text'] = df['text'].apply(preprocess)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['clean_text'])
    model = MultinomialNB()
    model.fit(X, df['label_num'])
    return model, vectorizer

model, vectorizer = load_model()

# Custom CSS for colored boxes
st.markdown("""
<style>
    .scam-box {
        background-color: #ffdddd;
        border-left: 5px solid #ff0000;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
        font-size: 18px;
    }
    .not-scam-box {
        background-color: #ddffdd;
        border-left: 5px solid #00aa00;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("📧 SPAM DETECTOR")
user_input = st.text_area("PASTE EMAIL TEXT HERE:", height=150)

if st.button("CHECK MESSAGE"):
    if user_input:
        clean_text = preprocess(user_input)
        text_vec = vectorizer.transform([clean_text])
        proba = model.predict_proba(text_vec)[0]
        is_scam = model.predict(text_vec)[0] == 1
        confidence = round(proba[1] * 100, 2)
        
        if is_scam:
            st.markdown(f"""
            <div class="scam-box">
                <h3>⚠️ SCAM DETECTED</h3>
                <p>Confidence: {confidence}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="not-scam-box">
                <h3>✅ LEGITIMATE MESSAGE</h3>
                <p>Confidence: {100-confidence}%</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to check")



       
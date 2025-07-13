# spamshieldai
📧 Email Spam Detector
A machine learning system that classifies emails as spam or legitimate with 90%+ accuracy.

✨ Key Features
1 Real-time predictions – Paste any email text for instant analysis
2 Visual feedback – Color-coded results (red for spam/green for ham)
3 Confidence scoring – See how confident the model is in its prediction
4 Lightweight – Runs locally without external APIs

🛠️ Built With
Python (Core language)

Streamlit (Web interface)

Scikit-learn (TF-IDF + Naive Bayes)

NLTK (Text tokenization & stopword removal)

🚀 Getting Started
Prerequisites

Python 3.8+

spam.csv file with label and text columns

Installation

bash
pip install -r requirements.txt
Usage

bash
streamlit run spam_detection.py
Then open http://localhost:8501 in your browser.

📊 How It Works
Text Cleaning: Removes punctuation, stopwords, and normalizes case

Feature Extraction: Converts text to numerical features using TF-IDF

Classification: Uses a pre-trained Naive Bayes model

Result Display: Shows prediction with confidence percentage

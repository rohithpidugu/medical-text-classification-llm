# src/preprocessing.py (UPDATED)

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- GUARANTEED NLTK DOWNLOADS ---
# This ensures the resources are available immediately upon import/use.
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet', quiet=True)
try:
    # This is the resource that was consistently missing: 'punkt'
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

# --- 1. Data Cleaning and Preprocessing Functions ---

def clean_text(text: str) -> str:
# ... (rest of the functions remain the same) ...

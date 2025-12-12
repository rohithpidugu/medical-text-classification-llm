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
    """Performs text cleaning: lowercasing, special character/number removal."""
    # 2. Convert text to lowercase
    text = text.lower() 
    # 3. Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text) 
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text: str) -> str:
    """Performs tokenization and lemmatization, and stop-word removal."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) # Stop-word removal
    
    # 4. Tokenize and lemmatize
    tokens = nltk.word_tokenize(text)
    
    # Apply filtering and lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 1
    ]
    return " ".join(lemmatized_tokens)

# --- 2. Feature Extraction Function (TF-IDF) ---

def create_tfidf_features(X_train: pd.Series, X_test: pd.Series) -> tuple:
    """
    Fits and transforms text data using TF-IDF vectorization.
    
    Uses reported parameters: max_features=3000 and ngram_range=(1, 2).
    
    """
    
    # TF-IDF calculation is the reported feature representation
    tfidf_params = {
        'max_features': 3000,
        'min_df': 5,
        'max_df': 0.8,
        'ngram_range': (1, 2), # Bigram inclusion
        'stop_words': 'english' # Stop-words handled explicitly here for vectorizer
    }
    
    vectorizer = TfidfVectorizer(**tfidf_params)
    
    # Fit on training data and transform both train/test
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Return vectorized data and the fitted vectorizer
    return X_train_vec, X_test_vec, vectorizer

# --- 3. Full Pipeline Function ---

def run_full_preprocessing_pipeline(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Applies cleaning and lemmatization to the specified text column."""
    print(f"Starting preprocessing on {text_column}...")
    
    # Apply cleaning and preprocessing
    df['clean_text'] = df[text_column].apply(clean_text)
    df['processed_text'] = df['clean_text'].apply(tokenize_and_lemmatize)
    
    print("Preprocessing complete.")
    return df

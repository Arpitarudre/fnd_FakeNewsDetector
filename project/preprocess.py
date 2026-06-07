# preprocess.py
"""
SPDX-License-Identifier: Apache-2.0
Fake News Detection: Natural Language Processing Preprocessing Module
Author: ML Engineer
Description: Provides clean, modular functions to preprocess raw news articles
             using NLTK (lowercasing, punctuation/number removal, stopword removal, lemmatization).
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download requisite NLTK corpora
# These will download automatically when the script is run
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab/english')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Define English Stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans and preprocesses the input text for the NLP models.
    Steps:
    1. Convert to lowercase
    2. Remove URL links
    3. Remove text within brackets/parentheses
    4. Remove punctuation, special characters, and numbers
    5. Tokenize text
    6. Filter out stopwords
    7. Apply WordNet Lemmatization
    8. Join back into a single string
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove URL links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove text inside brackets/parentheses
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    # 4. Remove XML/HTML tags
    text = re.sub(r'<.*?>+', '', text)
    
    # 5. Remove numbers and punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing digits
    
    # 6. Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Tokenize
    tokens = word_tokenize(text)
    
    # 8. Filter stopwords and apply Lemmatization
    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words and len(word) > 2:
            # WordNet lemmatizes nouns by default.
            lemma = lemmatizer.lemmatize(word, pos='v') # Lemmatize as verb
            lemma = lemmatizer.lemmatize(lemma, pos='n') # Lemmatize as noun
            cleaned_tokens.append(lemma)
            
    # 9. Return as single processed string
    return " ".join(cleaned_tokens)

if __name__ == "__main__":
    # Test the preprocessor
    sample_news = "BREAKING news! The president was seen eating a burger at 12 PM in NY: http://example.com/president-seen [EXCLUSIVE]"
    print("--- Original Text ---")
    print(sample_news)
    print("\n--- Cleaned Text ---")
    print(clean_text(sample_news))

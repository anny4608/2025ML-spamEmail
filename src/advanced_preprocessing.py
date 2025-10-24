"""Advanced text preprocessing for spam classification."""
import re
import string
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom text preprocessor that can be used in a scikit-learn pipeline."""
    
    def __init__(self, 
                 remove_urls=True,
                 remove_numbers=True,
                 remove_punctuation=True,
                 convert_lowercase=True,
                 remove_stopwords=True,
                 lemmatize=True):
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.convert_lowercase = convert_lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self._preprocess_text(text) for text in X]
    
    def _preprocess_text(self, text: str) -> str:
        """Apply all preprocessing steps to a single text."""
        
        # Convert to string in case we get something else
        text = str(text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        if self.convert_lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into text
        return ' '.join(tokens)

def clean_dataset(texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
    """Clean the dataset by removing duplicates and empty messages."""
    # Create a list of tuples (text, label)
    data = list(zip(texts, labels))
    
    # Remove duplicates while preserving order
    seen = set()
    cleaned_data = []
    for text, label in data:
        if text not in seen and text.strip():  # Check if text is not empty
            seen.add(text)
            cleaned_data.append((text, label))
    
    # Unzip the cleaned data
    cleaned_texts, cleaned_labels = zip(*cleaned_data)
    return list(cleaned_texts), list(cleaned_labels)

def extract_features(text: str) -> dict:
    """Extract additional features from text messages."""
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'contains_url': 1 if re.search(r'http\S+|www\S+|https\S+', text) else 0,
        'contains_number': 1 if re.search(r'\d+', text) else 0,
        'contains_currency': 1 if re.search(r'[$€£¥]', text) else 0,
        'contains_exclamation': 1 if '!' in text else 0,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / (len(text) or 1),
    }
    return features
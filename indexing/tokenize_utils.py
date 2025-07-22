import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from typing import List

# Uncomment if stopwords are not already downloaded
# nltk.download('stopwords')

LANGUAGE = "english"

# Initialize tools
nltk.data.path.append("../nltk_data")
from nltk.corpus import stopwords

stop_words = set(stopwords.words(LANGUAGE))
stemmer = SnowballStemmer(LANGUAGE)

def normalize_and_tokenize(text: str) -> List[str]:
    """
    Normalizes, tokenizes and stemms input text

    The function performs the following steps:
    1. Converts text to lowercase.
    2. Tokenizes text by extracting word-like tokens (alphanumeric sequences).
    3. Removes stopwords and short tokens (length <= 1).
    4. Applies stemming to reduce words to their root form.

    Args:
        text (str): The raw input string to be processed.

    Returns:
        List[str]: A list of normalized, stemmed tokens.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation, split into words (tokens)
    tokens = re.findall(r'\b\w+\b', text)

    # 3. Remove stopwords and short tokens, apply stemming
    cleaned_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    return cleaned_tokens

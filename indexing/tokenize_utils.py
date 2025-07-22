import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from typing import List

# Uncomment if stopwords are not already downloaded
# nltk.download('stopwords')

LANGUAGE = "english"

# Initialize tools
stop_words = set({'our', 'ourselves', 'if', "hasn't", 'before', 'my', 'while', 'his', 'once', 'over', 'itself', "she'll", 'don', 'where', 'm', "we're", "he's", "mustn't", 'on', "should've", "i'd", 'which', "doesn't", 'hadn', 'were', 've', 'its', 'again', 'above', 'won', "shouldn't", "we'd", 'their', 'an', 'myself', "she'd", 'until', "don't", 'theirs', 'will', 'for', 'can', 'ours', 'how', 'has', 'hasn', 'she', "you've", "she's", 'down', 'weren', 'when', "didn't", 'needn', 'so', 'why', 'out', 'd', 'this', 'from', 're', "won't", 'to', 'under', "shan't", 'wouldn', 'and', 'any', 'then', "isn't", 'than', "we've", 't', 'hers', 'they', "aren't", 'that', "you'd", 'or', 'been', 'being', 'shan', 'more', 'me', 'should', 'll', "he'd", 'most', 'both', 'wasn', 'it', 'was', "needn't", "you're", 'what', 'now', 'couldn', 'with', "i'll", 'whom', 'did', "wouldn't", 'do', 'other', 'the', "hadn't", 'nor', 'yourself', 'themselves', "weren't", 'during', 'is', 'herself', 'no', 'doing', "i've", 'up', 'because', "mightn't", 'doesn', 'o', 'at', 'didn', "they'd", 'mustn', 'just', 'each', 'of', "they're", 'few', 'her', 'isn', 'off', "that'll", 'by', 'against', 'himself', "they'll", 'through', 'yourselves', 'about', 'here', 'too', 'y', "it's", "you'll", 's', 'not', "he'll", 'ain', 'you', 'as', 'after', 'shouldn', 'only', 'i', 'does', 'some', "it'll", 'in', 'him', 'am', 'have', 'further', 'same', 'all', "i'm", 'yours', 'below', 'haven', 'having', 'mightn', 'into', 'be', 'aren', "we'll", 'such', "haven't", "they've", 'who', "wasn't", "it'd", 'them', 'those', 'between', 'very', 'a', "couldn't", 'these', 'your', 'we', 'are', 'own', 'but', 'ma', 'he', 'had', 'there'})
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

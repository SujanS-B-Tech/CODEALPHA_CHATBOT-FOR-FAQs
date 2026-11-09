"""
Text Preprocessing Module for SmartFAQ Chatbot.

Provides consistent text normalization, punctuation removal, whitespace trimming,
and token-level cleaning for both FAQ question corpus and incoming user queries.
"""

import re
import string
from typing import List, Optional


# Standard set of common conversational stopwords to optionally remove noise
COMMON_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now"
}


def clean_text(text: Optional[str]) -> str:
    """
    Cleans and normalizes text for NLP vectorization:
    - Handles None or non-string inputs safely.
    - Converts text to lowercase.
    - Strips URL/web addresses if present.
    - Replaces punctuation and special symbols with spaces.
    - Collapses multiple whitespace characters into single space.
    - Trims leading and trailing spaces.

    Parameters:
        text (str): Raw input string.

    Returns:
        str: Normalized, cleaned string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase normalization
    text = text.lower()

    # 2. Remove URLs if any
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Replace punctuation and non-alphanumeric characters with spaces
    # Keeps letters, numbers, and whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 4. Collapse multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: Optional[str]) -> List[str]:
    """
    Tokenizes normalized text into a list of words.

    Parameters:
        text (str): Input text.

    Returns:
        List[str]: List of individual word tokens.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Filters out common stopwords from a list of tokens.

    Parameters:
        tokens (List[str]): List of word tokens.

    Returns:
        List[str]: Filtered list of tokens.
    """
    return [token for token in tokens if token not in COMMON_STOPWORDS]


def preprocess_pipeline(text: Optional[str], remove_stops: bool = False) -> str:
    """
    Full text preprocessing pipeline for FAQ matching.
    Ensures identical treatment for stored FAQ questions and incoming user prompts.

    Parameters:
        text (str): Raw question string.
        remove_stops (bool): Whether to eliminate common stopwords.
                             Default False because TfidfVectorizer handles ngram stopwords.

    Returns:
        str: Fully processed string ready for TF-IDF vectorization.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    if remove_stops:
        tokens = tokenize(cleaned)
        filtered_tokens = remove_stopwords(tokens)
        return " ".join(filtered_tokens)

    return cleaned

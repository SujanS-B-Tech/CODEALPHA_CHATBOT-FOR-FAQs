"""
Comprehensive Test Suite for SmartFAQ Chatbot Engine and Preprocessing.

Tests:
- Exact FAQ matches
- Case insensitivity & punctuation noise
- Common keyword variations & paraphrasing
- Empty and whitespace queries
- Unrelated questions and threshold enforcement
- Missing or malformed CSV datasets
- Duplicate FAQ entries
- Empty datasets
- Text preprocessing pipeline
"""

import os
import tempfile
import pytest
import pandas as pd

from preprocessing import clean_text, tokenize, remove_stopwords, preprocess_pipeline
from chatbot import FAQChatbot


@pytest.fixture
def sample_csv():
    """Provides the path to the main faqs.csv dataset."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "faqs.csv")
    csv_path = os.path.abspath(csv_path)
    assert os.path.exists(csv_path), f"faqs.csv does not exist at {csv_path}"
    return csv_path


@pytest.fixture
def chatbot(sample_csv):
    """Provides a fitted FAQChatbot instance using the sample CSV."""
    bot = FAQChatbot(csv_path=sample_csv, similarity_threshold=0.30)
    return bot


# ============================================================================
# 1. PREPROCESSING TESTS
# ============================================================================

def test_clean_text_lowercase_and_punctuation():
    raw = "HOW DO I APPLY??? Please Help!!!"
    cleaned = clean_text(raw)
    assert cleaned == "how do i apply please help"
    assert "???" not in cleaned
    assert "!" not in cleaned


def test_clean_text_whitespace():
    raw = "   Can    I    take     leave?   \n\t "
    cleaned = clean_text(raw)
    assert cleaned == "can i take leave"


def test_clean_text_edge_cases():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   ") == ""
    assert clean_text("!@#$%^&*()") == ""
    assert clean_text(12345) == "12345"


def test_tokenize():
    tokens = tokenize("Internship duration is 4 weeks.")
    assert tokens == ["internship", "duration", "is", "4", "weeks"]
    assert tokenize("") == []
    assert tokenize(None) == []


def test_remove_stopwords():
    tokens = ["the", "duration", "of", "the", "internship"]
    filtered = remove_stopwords(tokens)
    assert "the" not in filtered
    assert "of" not in filtered
    assert "duration" in filtered
    assert "internship" in filtered


def test_preprocess_pipeline():
    result = preprocess_pipeline("  HOW to submit   PROJECT???  ")
    assert result == "how to submit project"


# ============================================================================
# 2. EXACT AND VARIATION MATCHING TESTS
# ============================================================================

def test_exact_matches(chatbot):
    exact_q = "How do I apply for the internship?"
    response = chatbot.get_response(exact_q)

    assert response["matched"] is True
    assert response["status"] == "SUCCESS"
    assert response["similarity_score"] >= 0.95
    assert "apply online" in response["answer"].lower()
    assert response["matched_question"] == exact_q


def test_case_insensitivity(chatbot):
    query = "HOW DO I APPLY FOR THE INTERNSHIP?"
    response = chatbot.get_response(query)

    assert response["matched"] is True
    assert response["similarity_score"] >= 0.90
    assert "apply online" in response["answer"].lower()


def test_extra_spaces_and_punctuation(chatbot):
    query = "   how... do   I   apply,,, for the internship???!!!   "
    response = chatbot.get_response(query)

    assert response["matched"] is True
    assert response["similarity_score"] >= 0.90


def test_keyword_variations_and_paraphrasing(chatbot):
    # Testing query with different phrasing but strong keyword overlap
    query = "what is the procedure for application submission"
    response = chatbot.get_response(query)

    assert response["matched"] is True
    assert response["similarity_score"] >= 0.30
    assert response["category"] == "Application Procedures" or "apply" in response["answer"].lower()


def test_resume_questions(chatbot):
    query = "What format should my student resume follow?"
    response = chatbot.get_response(query)

    assert response["matched"] is True
    assert response["category"] == "Resume Preparation"
    assert "resume" in response["answer"].lower()


def test_certificate_questions(chatbot):
    query = "Will I get a verified completion certificate?"
    response = chatbot.get_response(query)

    assert response["matched"] is True
    assert response["category"] == "Certificates"
    assert "certificate" in response["answer"].lower()


# ============================================================================
# 3. EDGE CASES: EMPTY, UNRELATED, AND OUT-OF-VOCABULARY
# ============================================================================

def test_empty_query(chatbot):
    response = chatbot.get_response("")
    assert response["matched"] is False
    assert response["status"] == "EMPTY_QUERY"
    assert response["similarity_score"] == 0.0


def test_whitespace_and_symbols_only(chatbot):
    response = chatbot.get_response("     !@#$%^&*()   ")
    assert response["matched"] is False
    assert response["status"] == "EMPTY_QUERY"
    assert response["similarity_score"] == 0.0


def test_unrelated_questions_fall_below_threshold(chatbot):
    # Completely unrelated question
    query = "What is the recipe for baking chocolate brownies in the microwave?"
    response = chatbot.get_response(query, threshold=0.30)

    assert response["matched"] is False
    assert response["status"] == "BELOW_THRESHOLD"
    assert response["similarity_score"] < 0.30
    assert "couldn't find a sufficiently matching answer" in response["answer"]


def test_custom_threshold_tuning(chatbot):
    # Query with moderate similarity
    query = "Can you tell me about the interview"
    
    # Low threshold should match
    res_low = chatbot.get_response(query, threshold=0.20)
    assert res_low["matched"] is True
    
    # High threshold should reject
    res_high = chatbot.get_response(query, threshold=0.85)
    assert res_high["matched"] is False
    assert res_high["status"] == "BELOW_THRESHOLD"


# ============================================================================
# 4. DATASET VALIDATION TESTS (MISSING, DUPLICATE, MALFORMED CSV)
# ============================================================================

def test_missing_csv_file():
    bot = FAQChatbot()
    with pytest.raises(FileNotFoundError):
        bot.load_data("non_existent_file_path_12345.csv")


def test_malformed_csv_missing_required_column():
    # Create temp CSV without "answer" column
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("question,category\nWhat is your name?,General\n")
        temp_path = f.name

    try:
        bot = FAQChatbot()
        with pytest.raises(ValueError) as excinfo:
            bot.load_data(temp_path)
        assert "missing required column 'answer'" in str(excinfo.value).lower()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_duplicate_questions_dropped():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(
            "id,category,question,answer\n"
            "1,General,How to apply?,Answer 1\n"
            "2,General,How to apply?,Answer 2 (duplicate)\n"
            "3,General,Where is office?,Answer 3\n"
        )
        temp_path = f.name

    try:
        bot = FAQChatbot(csv_path=temp_path)
        # Should drop the duplicate question
        assert len(bot.df) == 2
        assert bot.df.iloc[0]["answer"] == "Answer 1"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_missing_rows_dropped():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(
            "id,category,question,answer\n"
            "1,General,Valid Question?,Valid Answer\n"
            "2,General,,Missing Question Answer\n"
            "3,General,Missing Answer?, \n"
        )
        temp_path = f.name

    try:
        bot = FAQChatbot(csv_path=temp_path)
        assert len(bot.df) == 1
        assert bot.df.iloc[0]["question"] == "Valid Question?"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_empty_dataset_handling():
    bot = FAQChatbot()
    bot.fit(pd.DataFrame(columns=["question", "answer"]))
    res = bot.get_response("Hello?")
    assert res["matched"] is False
    assert res["status"] == "EMPTY_DATASET"
    assert "empty or not initialized" in res["answer"].lower()


def test_dataset_categories_and_samples(chatbot):
    categories = chatbot.get_categories()
    assert len(categories) == 10
    assert "Eligibility" in categories
    assert "Certificates" in categories
    assert "Project Submissions" in categories

    samples = chatbot.get_sample_questions(limit=5)
    assert len(samples) == 5

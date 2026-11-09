"""
Chatbot Engine Module for SmartFAQ.

Implements dataset loading and validation, TF-IDF vectorization,
and Cosine Similarity matching for intelligent FAQ retrieval.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_pipeline, clean_text


class FAQChatbot:
    """
    Intelligent FAQ Chatbot using TF-IDF and Cosine Similarity.
    """

    REQUIRED_COLUMNS = ["question", "answer"]
    DEFAULT_THRESHOLD = 0.30

    def __init__(
        self,
        csv_path: Optional[str] = None,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        """
        Initializes the FAQChatbot instance.

        Parameters:
            csv_path (str, optional): Path to the FAQ CSV dataset.
            similarity_threshold (float): Minimum cosine similarity to accept a match (default: 0.30).
            ngram_range (tuple): N-gram range for TfidfVectorizer (unigrams + bigrams).
        """
        self.csv_path = csv_path
        self.similarity_threshold = float(similarity_threshold)
        self.ngram_range = ngram_range

        self.df: pd.DataFrame = pd.DataFrame(columns=["id", "category", "question", "answer", "processed_question"])
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.is_fitted: bool = False

        if csv_path:
            self.load_and_fit(csv_path)

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Loads and validates FAQ data from a CSV file.
        Cleans missing values, removes duplicates, and validates structure.

        Parameters:
            csv_path (str): Path to CSV file.

        Returns:
            pd.DataFrame: Cleaned and validated DataFrame.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"FAQ dataset file not found at: {csv_path}")

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin1")

        # Validate required columns
        columns_lower = {col.lower(): col for col in df.columns}
        for req in self.REQUIRED_COLUMNS:
            if req not in columns_lower:
                raise ValueError(
                    f"Dataset is missing required column '{req}'. Found columns: {list(df.columns)}"
                )

        # Standardize column names
        rename_map = {columns_lower[req]: req for req in self.REQUIRED_COLUMNS}
        if "category" in columns_lower:
            rename_map[columns_lower["category"]] = "category"
        if "id" in columns_lower:
            rename_map[columns_lower["id"]] = "id"
        df = df.rename(columns=rename_map)

        if "category" not in df.columns:
            df["category"] = "General"
        if "id" not in df.columns:
            df["id"] = range(1, len(df) + 1)

        # Handle missing questions or answers
        initial_count = len(df)
        df = df.dropna(subset=["question", "answer"]).copy()

        # Clean string whitespace in raw data
        df["question"] = df["question"].astype(str).str.strip()
        df["answer"] = df["answer"].astype(str).str.strip()
        df["category"] = df["category"].astype(str).str.strip()

        # Drop empty strings
        df = df[(df["question"] != "") & (df["answer"] != "")].copy()

        # Drop duplicate questions (keep first occurrence)
        df = df.drop_duplicates(subset=["question"], keep="first").reset_index(drop=True)

        return df

    def fit(self, df: Optional[pd.DataFrame] = None) -> "FAQChatbot":
        """
        Fits the TF-IDF Vectorizer on preprocessed FAQ questions.

        Parameters:
            df (pd.DataFrame, optional): DataFrame containing FAQ entries.

        Returns:
            FAQChatbot: Self instance.
        """
        if df is not None:
            self.df = df.copy()

        if self.df.empty:
            self.is_fitted = False
            self.vectorizer = None
            self.tfidf_matrix = None
            return self

        # Apply NLP preprocessing pipeline to all FAQ questions
        self.df["processed_question"] = self.df["question"].apply(
            lambda q: preprocess_pipeline(q)
        )

        # Vectorizer configuration:
        # - unigrams & bigrams (1, 2) capture both single words and 2-word phrases
        # - sublinear_tf=True applies sublinear scaling (1 + log(tf)) to reduce weight of frequent terms
        # - stop_words='english' filters uninformative English stopwords
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            stop_words="english",
            sublinear_tf=True
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["processed_question"])
        self.is_fitted = True
        return self

    def load_and_fit(self, csv_path: str) -> "FAQChatbot":
        """
        Convenience method to load CSV and fit vectorizer in one call.

        Parameters:
            csv_path (str): Path to CSV.

        Returns:
            FAQChatbot: Self instance.
        """
        self.csv_path = csv_path
        df = self.load_data(csv_path)
        return self.fit(df)

    def get_response(
        self,
        user_query: str,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a user question and retrieves the most similar FAQ answer.

        Parameters:
            user_query (str): The raw text question entered by the user.
            threshold (float, optional): Custom similarity threshold. Defaults to self.similarity_threshold.

        Returns:
            dict: {
                "matched": bool,
                "answer": str,
                "matched_question": Optional[str],
                "category": Optional[str],
                "similarity_score": float,
                "top_candidates": List[Dict],
                "status": str
            }
        """
        if threshold is None:
            threshold = self.similarity_threshold

        # 1. Edge Case: Empty or invalid input
        cleaned_query = clean_text(user_query)
        if not cleaned_query:
            return {
                "matched": False,
                "answer": "Please ask a question! Type your inquiry about internship eligibility, applications, documents, or support.",
                "matched_question": None,
                "category": None,
                "similarity_score": 0.0,
                "top_candidates": [],
                "status": "EMPTY_QUERY"
            }

        # 2. Edge Case: Dataset not loaded or empty
        if not self.is_fitted or self.df.empty or self.vectorizer is None or self.tfidf_matrix is None:
            return {
                "matched": False,
                "answer": "The FAQ knowledge base is currently empty or not initialized. Please load a valid FAQ dataset.",
                "matched_question": None,
                "category": None,
                "similarity_score": 0.0,
                "top_candidates": [],
                "status": "EMPTY_DATASET"
            }

        # 3. Preprocess user query using identical pipeline
        processed_query = preprocess_pipeline(user_query)
        query_vector = self.vectorizer.transform([processed_query])

        # 4. Compute Cosine Similarity between query vector and all FAQ vectors
        # Result is a 1 x N array of similarity values between 0.0 and 1.0
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        # Extract top 3 candidates for transparency and debugging
        top_indices = np.argsort(similarities)[::-1][:3]
        top_candidates = []
        for idx in top_indices:
            top_candidates.append({
                "question": self.df.iloc[idx]["question"],
                "score": round(float(similarities[idx]), 4),
                "category": self.df.iloc[idx].get("category", "General")
            })

        # 5. Threshold comparison
        if best_score >= threshold:
            best_row = self.df.iloc[best_index]
            return {
                "matched": True,
                "answer": best_row["answer"],
                "matched_question": best_row["question"],
                "category": best_row.get("category", "General"),
                "similarity_score": round(best_score, 4),
                "top_candidates": top_candidates,
                "status": "SUCCESS"
            }
        else:
            # Score below threshold - return polite fallback
            fallback_msg = (
                f"I'm sorry, I couldn't find a sufficiently matching answer in our FAQ knowledge base "
                f"(Highest similarity: {best_score:.2f}, required threshold: {threshold:.2f}).\n\n"
                "**Tips to get better answers:**\n"
                "- Try rephrasing with key internship terms (e.g., 'resume format', 'certificates', 'apply deadline').\n"
                "- Check the suggested questions in the sidebar or prompt chips above."
            )
            return {
                "matched": False,
                "answer": fallback_msg,
                "matched_question": self.df.iloc[best_index]["question"] if best_score > 0 else None,
                "category": self.df.iloc[best_index].get("category", "General") if best_score > 0 else None,
                "similarity_score": round(best_score, 4),
                "top_candidates": top_candidates,
                "status": "BELOW_THRESHOLD"
            }

    def get_categories(self) -> List[str]:
        """Returns list of unique FAQ categories."""
        if self.df.empty or "category" not in self.df.columns:
            return []
        return sorted(self.df["category"].unique().tolist())

    def get_sample_questions(self, limit: int = 5) -> List[str]:
        """Returns a diverse sample of questions from different categories."""
        if self.df.empty:
            return []
        # Sample one from each category up to limit
        grouped = self.df.groupby("category", as_index=False).first()
        sample = grouped["question"].head(limit).tolist()
        return sample

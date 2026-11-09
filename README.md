# SmartFAQ — Intelligent FAQ Assistant

An intelligent, natural language processing (NLP) FAQ chatbot web application built with **Python**, **Streamlit**, and **Scikit-learn**. The application processes student inquiries using **TF-IDF vectorization** and retrieves the most relevant answers via **Cosine Similarity** from a curated 55+ question dataset.

---

## 📌 Project Overview

**SmartFAQ** is designed for internship programs and academic student-support portals. Instead of relying on rigid, hardcoded `if-else` matching, it transforms text into numerical vectors and evaluates mathematical similarity in multi-dimensional vector space.

### Key Highlights
- **10 Dedicated Categories**: Eligibility, Application Procedures, Required Documents, Resume Preparation, Interview Preparation, Internship Duration, Certificates, Attendance, Project Submissions, and Technical Support.
- **Real-Time Similarity Scoring**: Displays cosine confidence scores alongside answers.
- **Configurable Fallback Threshold**: Dynamically filters out low-confidence answers (default: `0.30`).
- **Interactive Streamlit Web UI**: Includes chat bubbles, clickable sample prompts, sidebar analytics, expandable NLP matching details, and instant chat reset.
- **Zero Paid APIs**: Runs 100% locally with high performance and zero external API dependencies.

---

## 🧠 NLP Algorithm Explained (For Your Internship Supervisor)

### 1. What is TF-IDF?
**TF-IDF** stands for **Term Frequency – Inverse Document Frequency**. It converts text strings into numerical vectors that represent how informative each word is within a document:

1. **Term Frequency ($\text{TF}$)**: Measures how frequently a word appears in a specific question.
   $$\text{TF}(t, d) = \frac{\text{Count of word } t \text{ in question } d}{\text{Total words in question } d}$$
2. **Inverse Document Frequency ($\text{IDF}$)**: Measures how rare or unique the word is across the entire dataset. Common conversational words like *"what"*, *"is"*, or *"the"* receive low weights, while distinctive terms like *"transcript"*, *"certificate"*, or *"reapply"* receive high weights.
   $$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
3. **$\text{TF-IDF}$ Score**:
   $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

In SmartFAQ, we use **unigrams and bigrams (`ngram_range=(1, 2)`)** with sublinear term-frequency scaling so that two-word phrases (e.g., *"project submission"*, *"resume format"*) are treated as distinct features.

---

### 2. What is Cosine Similarity?
Once both the FAQ dataset and the incoming user query are converted into numerical TF-IDF vectors, we measure how closely they point in the same direction in vector space.

$$\text{Cosine Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$

- **Score of 1.0**: Perfect lexical match.
- **Score of 0.40 – 0.90**: Strong to moderate match with similar terminology.
- **Score < 0.30**: Weak or unrelated match; triggers our polite fallback response.

---

### 3. TF-IDF vs. Large Language Models (LLMs)

| Feature | TF-IDF & Cosine Similarity (SmartFAQ) | Large Language Models (e.g., GPT, Claude) |
| :--- | :--- | :--- |
| **Matching Nature** | Lexical & Statistical pattern matching | Deep semantic & generative reasoning |
| **Cost & Compute** | 100% free, lightweight, instantaneous CPU computation | Heavy GPU usage or paid cloud API fees |
| **Hallucination Risk** | **0%** (Always returns verified human-authored answers) | Non-zero (Risk of generating inaccurate answers) |
| **Data Privacy** | Completely offline & secure on local machine | Sends sensitive queries to third-party servers |
| **Synonym Handling** | Limited to tokens present in the vocabulary | Understands synonymous phrasing natively |

> [!NOTE]
> TF-IDF is deterministic and safe for enterprise compliance because it strictly returns verified corporate/institution answers without generative hallucination.

---

## 🏗️ System Architecture

```
[ User Input Query ]
         │
         ▼
[ Text Preprocessing: lowercase, punctuation cleaning, whitespace normalization ]
         │
         ▼
[ TF-IDF Vectorizer Transform (Fitted on FAQ questions) ]
         │
         ▼
[ Cosine Similarity Engine ] ◄── [ Pre-computed TF-IDF Matrix of FAQs ]
         │
         ▼
[ Best Match Selection: argmax(cosine_sim) ]
         │
    ┌────┴───────────────────────────┐
    │ Score >= Threshold (e.g., 0.30)?│
   YES                              NO
    │                                │
    ▼                                ▼
[ Return Verified FAQ Answer ]     [ Polite Fallback & Suggestions ]
    │                                │
    └──────────────┬─────────────────┘
                   ▼
[ Streamlit Modern Chat Interface ]
  - User & Assistant Chat Bubbles
  - Confidence Score Badge (High / Moderate / Low)
  - Expandable Matching Breakdown & Top-3 Candidates
```

---

## 📁 Project Structure

```
smartfaq-chatbot/
│
├── app.py                   # Streamlit web application & chat UI
├── chatbot.py               # FAQChatbot engine (data loading, TF-IDF, similarity)
├── preprocessing.py         # Text normalization and token-level cleaning pipeline
├── faqs.csv                 # 55+ Curated FAQ questions, categories, and answers
├── start.bat                # One-click Windows launcher (auto setup & run)
├── requirements.txt         # Project dependencies
├── README.md                # Comprehensive documentation & setup instructions
├── .gitignore               # Git ignore rules for virtualenvs and caches
└── tests/
    └── test_chatbot.py      # Automated Pytest test suite covering all edge cases
```

---

## 🚀 Setup & Launch Instructions (Windows)

### ⚡ Quick Start: One-Click Launch
Simply **double-click `start.bat`** in File Explorer (or run `.\start.bat` in PowerShell). 
It will automatically:
1. Verify Python installation.
2. Create the `venv` virtual environment if it doesn't already exist.
3. Install and upgrade all dependencies from `requirements.txt`.
4. Launch the Streamlit application and open your browser!

---

### Manual Setup via PowerShell
If you prefer running commands manually:

#### 1. Open PowerShell and Navigate to the Project Directory
```powershell
cd "d:\PROJECTS\Projects\CODEALPHAINTERN\CHATBOT FOR FAQs"
```

### 2. Create and Activate a Virtual Environment
```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Automated Unit & Integration Tests
```powershell
pytest -v tests/test_chatbot.py
```
All tests verify exact matching, punctuation noise handling, keyword variations, empty inputs, unrelated queries, and dataset validation.

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Streamlit will automatically open your default browser at `http://localhost:8501`.

---

## 📊 Dataset Breakdown (`faqs.csv`)

The dataset contains **55 questions and answers** across 10 vital domains:

1. **Eligibility (6 FAQs)**: Who can apply, non-tech background, CGPA criteria, international students, 1st-year students, reapplying.
2. **Application Procedures (6 FAQs)**: Application steps, application fees, cohort deadlines, status tracking, multi-domain policy, error correction.
3. **Required Documents (6 FAQs)**: ID proofs, No Objection Certificate (NOC), file formats (PDF < 5MB), LORs, mark sheets, scanned copies.
4. **Resume Preparation (5 FAQs)**: ATS formatting, project inclusions, key programming skills, extracurricular balance, single-page rule.
5. **Interview Preparation (5 FAQs)**: Rounds format, technical preparation, live coding assessments, STAR behavioral method, unknown questions.
6. **Internship Duration (5 FAQs)**: 4/8/12 weeks duration, full-time vs part-time, project extensions, exam leaves, early completion.
7. **Certificates (6 FAQs)**: Verifiable certificates, QR verification portal, LOR criteria, timeline (7-10 days), academic accreditation, name corrections.
8. **Attendance (5 FAQs)**: Tracking via sprints/Discord/Git, 80% threshold, missed sessions, weekend rules, exam adjustments.
9. **Project Submissions (6 FAQs)**: Submission requirements (GitHub, demo, video), late policies, 5-point evaluation criteria, team tracks, topic selection, repo guidelines.
10. **Technical Support (5 FAQs)**: Issue ticketing, environment setup support, cloud sandbox credits, supported OS (Windows/Mac/Linux), access recovery.

### How to Add or Modify FAQs
You can open `faqs.csv` in Excel, VS Code, or Notepad and add rows:
```csv
56,Certificates,Can I get a physical hard-copy certificate?,Yes, physical hard-copy certificates can be requested for postal dispatch by paying courier charges.
```
SmartFAQ will automatically ingest and re-vectorize the new questions upon restarting or clearing chat!

---

## 🧪 Automated Test Coverage

The test suite in `tests/test_chatbot.py` verifies:
- `test_exact_matches`: Exact questions produce similarity $\approx 1.0$ and return the exact answer.
- `test_case_insensitivity`: Uppercase, lowercase, and mixed cases yield identical matches.
- `test_extra_spaces_and_punctuation`: Queries with noise, punctuation, and extra whitespace are normalized.
- `test_keyword_variations_and_paraphrasing`: Rephrased queries match the target category.
- `test_empty_query` & `test_whitespace_and_symbols_only`: Graceful notification without crashes.
- `test_unrelated_questions_fall_below_threshold`: Unrelated queries (e.g., cooking recipes) trigger fallback.
- `test_custom_threshold_tuning`: Threshold adjustments dynamically accept or reject borderline queries.
- `test_missing_csv_file` & `test_malformed_csv_missing_required_column`: Proper exceptions raised on invalid datasets.
- `test_duplicate_questions_dropped`: Duplicate questions are cleanly filtered.
- `test_missing_rows_dropped`: Empty rows are removed automatically.
- `test_empty_dataset_handling`: Graceful response when dataset has no rows.

---

## ⚠️ Known Limitations & Future Enhancements

1. **Lexical Matching Limitation**: TF-IDF relies on token overlap. If a user asks *"How much currency must I remit to enroll?"* instead of *"Is there an application fee?"*, TF-IDF may yield a low score due to absent vocabulary overlap.
   - *Future Enhancement*: Implement dense word embeddings (e.g., `SentenceTransformers` or `Word2Vec`) to capture semantic synonyms.
2. **Spelling Mistakes**: Significant typos (e.g., *"documnts"* instead of *"documents"*) can lower TF-IDF token weights.
   - *Future Enhancement*: Add Levenshtein distance or SymSpell auto-correction before vectorization.
3. **Multi-turn Context**: SmartFAQ evaluates each query independently; it does not carry over pronoun references (e.g., *"Tell me more about it"*).

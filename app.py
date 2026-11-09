"""
SmartFAQ — Intelligent FAQ Assistant
Streamlit Web Application powered by TF-IDF & Cosine Similarity NLP matching.
"""

import os
import streamlit as st
import pandas as pd
from chatbot import FAQChatbot

# Page configuration
st.set_page_config(
    page_title="SmartFAQ — Intelligent FAQ Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, polished aesthetics
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Header hero styling */
    .hero-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    .badge-nlp {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 10px;
    }
    
    /* Confidence score badges */
    .score-badge-high {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .score-badge-med {
        background-color: rgba(234, 179, 8, 0.2);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .score-badge-low {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Card styling */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }

    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_faq_engine(csv_path: str) -> FAQChatbot:
    """Initializes and caches the FAQChatbot instance."""
    bot = FAQChatbot(csv_path=csv_path)
    return bot


def initialize_session():
    """Initializes session state variables for chat history and settings."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! Welcome to SmartFAQ.**\n\n"
                    "I am your intelligent assistant for internship questions, documents, "
                    "application procedures, project submissions, and student support.\n\n"
                    "Ask me anything below or click one of the suggested prompts!"
                ),
                "matched_question": None,
                "category": None,
                "similarity_score": None,
                "matched": True,
                "top_candidates": []
            }
        ]
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def render_sidebar(bot: FAQChatbot):
    """Renders configuration, stats, and NLP explainer in the sidebar."""
    with st.sidebar:
        st.markdown("### ⚙️ Engine Settings")
        
        # Similarity Threshold Slider
        threshold = st.slider(
            "Cosine Similarity Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
            help=(
                "Minimum cosine similarity score required for an FAQ match. "
                "Higher values require closer wording; lower values allow more flexible matches."
            )
        )
        
        st.caption(f"🎯 Current threshold: **{threshold:.2f}** (Recommended: 0.25 - 0.35)")

        st.divider()

        # Knowledge Base Statistics
        st.markdown("### 📊 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total FAQs", len(bot.df))
        with col2:
            st.metric("Categories", len(bot.get_categories()))

        if bot.vectorizer is not None:
            vocab_size = len(bot.vectorizer.vocabulary_)
            st.caption(f"🔤 Vocabulary Features: **{vocab_size} tokens** (Unigrams & Bigrams)")

        st.divider()

        # Chat Control
        if st.button("🗑️ Clear Conversation", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            initialize_session()
            st.rerun()

        st.divider()

        # Educational NLP explanation
        with st.expander("ℹ️ About the NLP Algorithm"):
            st.markdown("""
            **How SmartFAQ Works:**
            1. **Text Preprocessing**: Both dataset questions and your inputs are lowercased, stripped of noise/punctuation, and cleaned.
            2. **TF-IDF Vectorization**: Words are converted into numerical weight vectors based on **Term Frequency** (how often a term appears in a question) and **Inverse Document Frequency** (how unique that term is across the entire FAQ set).
            3. **Cosine Similarity**: We calculate the cosine of the angle between your query vector and all FAQ vectors in high-dimensional space:
            $$\\text{Cosine Sim} = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$$
            4. **Threshold Matching**: If the maximum score meets or exceeds the threshold, the matching answer is returned; otherwise, an intelligent fallback is triggered.
            
            *Note: TF-IDF matches statistical lexical patterns rather than neural conceptual reasoning.*
            """)

        # Dataset Explorer
        with st.expander("📚 Browse FAQ Topics"):
            selected_cat = st.selectbox("Filter by Category", ["All"] + bot.get_categories())
            if selected_cat == "All":
                display_df = bot.df[["id", "category", "question"]]
            else:
                display_df = bot.df[bot.df["category"] == selected_cat][["id", "category", "question"]]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    return threshold


def get_confidence_badge_html(score: float, threshold: float) -> str:
    """Returns HTML markup for confidence level badge."""
    if score >= 0.60:
        cls_name = "score-badge-high"
        label = "High Confidence"
    elif score >= threshold:
        cls_name = "score-badge-med"
        label = "Moderate Match"
    else:
        cls_name = "score-badge-low"
        label = "Below Threshold"
    return f'<span class="{cls_name}">{score:.2f} ({label})</span>'


def main():
    initialize_session()

    # Determine CSV file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "faqs.csv")

    try:
        bot = load_faq_engine(csv_path)
    except Exception as e:
        st.error(f"Error loading FAQ database from `{csv_path}`: {e}")
        return

    # Render Sidebar and retrieve threshold setting
    threshold = render_sidebar(bot)

    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">
            SmartFAQ <span class="badge-nlp">NLP TF-IDF & Cosine Similarity</span>
        </h1>
        <p class="hero-subtitle">
            Instant, automated answers to student & internship inquiries with real-time vector similarity scoring.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggested Prompts Bar
    st.markdown("##### 💡 Suggested Questions")
    sample_prompts = [
        "How do I apply for the internship?",
        "What documents are required?",
        "What should I include on my resume?",
        "Will I receive a verified certificate?",
        "How do I submit my final project?"
    ]
    
    cols = st.columns(len(sample_prompts))
    for idx, prompt in enumerate(sample_prompts):
        if cols[idx].button(prompt, key=f"sugg_{idx}", use_container_width=True):
            st.session_state.pending_question = prompt

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # Render Conversation History
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
            st.markdown(msg["content"])
            
            # If assistant message has matching metadata, render expandable details
            if role == "assistant" and msg.get("similarity_score") is not None:
                score = msg["similarity_score"]
                matched_q = msg.get("matched_question")
                category = msg.get("category")
                top_candidates = msg.get("top_candidates", [])
                
                with st.expander("🔍 Matching Details & Score"):
                    badge_html = get_confidence_badge_html(score, threshold)
                    st.markdown(f"**Similarity Score:** {badge_html}", unsafe_allow_html=True)
                    if matched_q:
                        st.markdown(f"**Matched Question:** *\"{matched_q}\"*")
                    if category:
                        st.markdown(f"**Category:** `{category}`")
                        
                    if top_candidates:
                        st.markdown("**Top Candidate Matches:**")
                        cand_df = pd.DataFrame(top_candidates)
                        st.dataframe(
                            cand_df.rename(columns={"question": "FAQ Question", "score": "Cosine Similarity", "category": "Category"}),
                            hide_index=True,
                            use_container_width=True
                        )

    # Handle input: Check either pending prompt from button click or chat_input
    user_input = None
    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None
    else:
        chat_val = st.chat_input("Ask a question about internships, requirements, duration, projects...")
        if chat_val:
            user_input = chat_val

    # Process new user input
    if user_input:
        # Add User message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Generate response using FAQ chatbot
        response_data = bot.get_response(user_input, threshold=threshold)

        # Append Assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["answer"],
            "matched_question": response_data["matched_question"],
            "category": response_data["category"],
            "similarity_score": response_data["similarity_score"],
            "matched": response_data["matched"],
            "top_candidates": response_data["top_candidates"]
        })

        st.rerun()


if __name__ == "__main__":
    main()

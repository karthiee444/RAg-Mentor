import os
import streamlit as st
import numpy as np
from pypdf import PdfReader
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb

# Try importing Groq client
try:
    from groq import Groq
except ImportError:
    Groq = None


# Config
SIMILARITY_THRESHOLD = 0.20
TOP_K = 3


# Utility Functions

def load_api_key() -> str:
    """Load the GROQ API key from environment or Hugging Face token fallback."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            from huggingface_hub import HfFolder
            api_key = HfFolder.get_token()
        except Exception:
            pass
    return api_key


def setup_groq() -> Groq:
    """Initialize Groq client with API key."""
    api_key = load_api_key()
    if not api_key:
        st.error("❌ Missing GROQ_API_KEY in environment or Hugging Face secrets.")
        return None
    if Groq is None:
        st.error("❌ Groq library not installed. Please add `groq` to requirements.txt.")
        return None
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return None


@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache the embedding model."""
    return SentenceTransformer(model_name)


def pdf_to_chunks(uploaded_file, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Convert PDF to overlapping text chunks."""
    try:
        reader = PdfReader(uploaded_file)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text.strip():
            continue

        words = text.split()
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i:i + chunk_size])
            if chunk_text.strip():
                chunks.append({
                    "page_number": page_num,
                    "text": chunk_text
                })
    return chunks


def create_vector_database(chunks: List[Dict], embedding_model: SentenceTransformer) -> str:
    """Create a new ChromaDB collection with embeddings and return its name."""
    if not chunks:
        st.error("No text chunks extracted from PDF.")
        return None

    client = chromadb.Client()
    collection_name = f"pdf_chunks_{np.random.randint(10000)}"

    try:
        collection = client.create_collection(collection_name)
    except Exception as e:
        st.error(f"Error creating collection: {e}")
        return None

    texts = [c["text"] for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    # Encode in batches for safety
    embeddings = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = embedding_model.encode(batch)
        embeddings.extend(emb.tolist() if hasattr(emb, 'tolist') else list(map(list, emb)))

    try:
        collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=chunks
        )
    except Exception as e:
        st.error(f"Error adding embeddings: {e}")
        return None

    # Store only the collection name (not object) in session_state
    st.session_state.collection_name = collection_name
    # Also store a simple flag in vector_db for UI readiness
    st.session_state.vector_db = collection_name
    return collection_name


def query_vector_database(query: str, embedding_model: SentenceTransformer,
                          top_k: int = TOP_K) -> List[Dict]:
    """Query ChromaDB for relevant chunks."""
    if "collection_name" not in st.session_state:
        st.error("No active collection found. Upload and process a PDF first.")
        return []

    try:
        client = chromadb.Client()
        collection = client.get_collection(st.session_state.collection_name)
    except Exception as e:
        st.error(f"Error accessing collection: {e}")
        return []

    try:
        query_embedding = embedding_model.encode([query]).tolist()
    except Exception as e:
        st.error(f"Error encoding query: {e}")
        return []

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
    except Exception as e:
        st.error(f"Error querying database: {e}")
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0] if "distances" in results else []

    relevant_chunks = []
    for i, doc in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) else {}
        distance = dists[i] if i < len(dists) else None

        if distance is None:
            similarity = 1.0
        elif isinstance(distance, (int, float)) and distance <= 1:
            similarity = max(0, 1 - distance)
        else:
            try:
                similarity = float(distance)
            except Exception:
                similarity = 0.0

        relevant_chunks.append({
            "text": doc,
            "page_number": meta.get("page_number", "N/A"),
            "similarity": similarity
        })

    return relevant_chunks


def generate_answer_with_groq(client, query: str, relevant_chunks: List[Dict]) -> str:
    """Generate answer from Groq LLM using retrieved context."""
    try:
        context_parts = [f"[Page {c['page_number']}]: {c['text']}" for c in relevant_chunks]
        context = "\n\n".join(context_parts) if context_parts else ""

        prompt = f"""Based ONLY on the following context from a PDF document, answer the user's question.

Context:
{context}

Question: {query}

Instructions:
- Answer using ONLY the information provided in the context above
- If the context does not contain enough information to answer the question, reply exactly: ❌ Insufficient evidence
- Always include page citations in your answer using the format [Page X]
- Be accurate and concise
- Do not add information not present in the context

Answer:"""

        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            chat_resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a strict assistant that only uses provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
        else:
            # Fallback generic call
            chat_resp = client.create(prompt=prompt, max_tokens=500)

        if hasattr(chat_resp, "choices"):
            return chat_resp.choices[0].message.content
        elif isinstance(chat_resp, dict):
            choices = chat_resp.get("choices") or []
            if choices:
                return choices[0].get("message", {}).get("content") \
                       or choices[0].get("text") \
                       or str(choices[0])
        return str(chat_resp)

    except Exception as e:
        return f"Error generating answer: {e}"



# STREAMLIT UI

# STREAMLIT UI

def main():
    """Main Streamlit application."""

    # Page configuration with wide layout for centered design
    st.set_page_config(
        page_title="PageMentor",
        page_icon="📚",
        layout="wide"
    )

    # Custom CSS for professional styling and centered layout
    st.markdown("""
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            color: #222; /* Default text color for light background */
            background-color: #f9f9f9; /* Light background */
        }

        /* Center main container */
        .main > div {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }

        /* Header styling */
        .header-container {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header-title {
            color: #fff;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .header-subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
        }

        /* Answer box */
        .answer-box {
            background-color: #ffffff;
            color: #222;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid #667eea;
        }

        /* Source cards */
        .source-card {
            background-color: #f4f4f8;
            color: #222;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 3px solid #764ba2;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102,126,234,0.4);
        }

        /* Uploaded file box */
        .uploadedFile {
            background-color: #fff;
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid #e0e0e0;
        }

        /* Inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            padding: 0.75rem;
            color: #222;
            background-color: #fff;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102,126,234,0.1);
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            margin-top: 3rem;
            border-top: 1px solid #ddd;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header Section with gradient background
    st.markdown("""
        <div class="header-container">
            <div class="header-title">📚 PageMentor</div>
            <div class="header-subtitle">Book-based AI Tutor - Learn from any PDF document</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Session state init
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = None

    # Load embedding model if not loaded
    if st.session_state.embedding_model is None:
        with st.spinner("🔄 Loading AI models..."):
            st.session_state.embedding_model = load_embedding_model()

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.container():
            st.markdown("### 📄 Upload Your Document")
            st.markdown("*Select a PDF file to start learning*")

            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Upload any PDF document - textbooks, research papers, articles, etc.",
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                st.info(f"📎 **File:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

                if st.button("🚀 Process Document", use_container_width=True):
                    # attempt best-effort cleanup of prior collection
                    try:
                        old_name = st.session_state.get("collection_name")
                        if old_name:
                            client_tmp = chromadb.Client()
                            if hasattr(client_tmp, "delete_collection"):
                                try:
                                    client_tmp.delete_collection(old_name)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # reset state
                    st.session_state.vector_db = None
                    st.session_state.collection_name = None
                    st.session_state.processed_file = None

                    # process file
                    with st.spinner("📖 Reading and analyzing your document..."):
                        chunks = pdf_to_chunks(uploaded_file)

                        if not chunks:
                            st.error("❌ Failed to extract any text from the uploaded PDF.")
                        else:
                            total_pages = len({c['page_number'] for c in chunks})
                            st.success(f"✅ Successfully processed **{total_pages} pages**")
                            st.info(f"📝 Created **{len(chunks)}** searchable text segments")

                            # Create vector database
                            if st.session_state.embedding_model:
                                with st.spinner("🧠 Building knowledge base..."):
                                    collection_name = create_vector_database(chunks, st.session_state.embedding_model)
                                    if collection_name:
                                        st.session_state.processed_file = uploaded_file.name
                                        st.success("✅ **Ready to answer your questions!**")
                                        st.balloons()
                                    else:
                                        st.error("❌ Failed to create knowledge base")
                            else:
                                st.error("❌ AI model not available")

    # Question answering section
    if st.session_state.vector_db is not None:
        st.markdown("---")
        st.markdown("### 💬 Ask Your Questions")

        if st.session_state.processed_file:
            st.markdown(f"*Currently learning from: **{st.session_state.processed_file}***")

        with st.form(key="question_form"):
            question = st.text_input(
                "What would you like to know?",
                placeholder="e.g., What is the main topic? Summarize chapter 3. Explain the key concepts.",
                help="Ask any question about the content of your document",
                label_visibility="collapsed"
            )

            submit_button = st.form_submit_button(
                "🔍 Get Answer",
                use_container_width=True
            )

        if submit_button and question.strip():
            with st.spinner("🤔 Thinking..."):
                # Query vector database
                embedding_model = st.session_state.embedding_model
                relevant_chunks = query_vector_database(
                    question,
                    embedding_model,
                    top_k=TOP_K
                )

                # Filter by similarity threshold
                relevant_chunks = [c for c in relevant_chunks if c.get('similarity', 0) >= SIMILARITY_THRESHOLD]

            # After spinner
            if not relevant_chunks:
                st.warning("❌ No sufficiently relevant passages found (increase threshold or rephrase question).")
            else:
                # Generate answer
                client = setup_groq()
                if not client:
                    st.error("❌ LLM not configured. Check GROQ_API_KEY and that 'groq' is installed.")
                else:
                    answer = generate_answer_with_groq(client, question, relevant_chunks)

                    # Display answer in chat bubble style
                    st.markdown("#### 🎯 Answer")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                    # Display sources in a clean format
                    st.markdown("#### 📚 Top Sources")
                    st.markdown("*Most relevant passages from your document:*")

                    for i, chunk in enumerate(relevant_chunks, 1):
                        with st.expander(
                            f"**Source {i}** | 📄 Page {chunk['page_number']} | "
                            f"🎯 Relevance: {chunk['similarity']*100:.0f}%"
                        ):
                            st.markdown(f'<div class="source-card">{chunk["text"][:500]}...</div>',
                                        unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background-color: white; border-radius: 15px; margin: 2rem 0;'>
            <h3>👋 Welcome to PageMentor!</h3>
            <p style='color: #666; font-size: 1.1rem;'>Upload a PDF document above to start your learning journey.</p>
            <p style='color: #999;'>Support for textbooks, research papers, articles, and more!</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit | Powered by Hugging Face | © 2025 PageMentor</p>
        <p style='font-size: 0.9rem; color: #999;'>Transform any document into your personal tutor</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

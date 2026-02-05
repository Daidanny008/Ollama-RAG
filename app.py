import os
import uuid
import tempfile
import streamlit as st

from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Fast PDF Chat", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat" not in st.session_state:
    st.session_state.chat = []

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def load_llm():
    return Ollama(
        model="llama3:8b",
        context_window=8192,      # plenty for RAG
        request_timeout=60.0,     # much faster now
        temperature=0.1,
    )

@st.cache_resource
def load_embed_model():
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )


@st.cache_resource
def build_index(file_bytes: bytes, filename: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, filename)
        with open(path, "wb") as f:
            f.write(file_bytes)

        docs = SimpleDirectoryReader(
            input_dir=tmp,
            required_exts=[".pdf"]
        ).load_data()

    embed_model = load_embed_model()
    return VectorStoreIndex.from_documents(
        docs,
        embed_model=embed_model,
        show_progress=False
    )


# -----------------------------
# Sidebar: upload PDF
# -----------------------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded:
        st.info("Indexing document…")
        index = build_index(uploaded.getvalue(), uploaded.name)

        llm = load_llm()
        qe = index.as_query_engine(
            llm=llm,
            streaming=True,
            similarity_top_k=4
        )

        prompt = PromptTemplate(
        """You are a helpful assistant.

        If the question is about the uploaded document, use the context below.
        If the question is unrelated, answer normally using your general knowledge.
        If you truly do not know the answer, say "I don't know".

        Context:
        {context_str}

        Question: {query_str}
        Answer:"""
        )


        qe.update_prompts({
            "response_synthesizer:text_qa_template": prompt
        })

        st.session_state.query_engine = qe
        st.success("Ready to chat!")


# -----------------------------
# Main UI
# -----------------------------
st.title("📄 Fast PDF Chat")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if st.session_state.query_engine:
    user_input = st.chat_input("Ask something about the document…")

    if user_input:
        st.session_state.chat.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_text = ""

            response = st.session_state.query_engine.query(user_input)

            for token in response.response_gen:
                response_text += token
                placeholder.markdown(response_text + "▌")

            placeholder.markdown(response_text)

        st.session_state.chat.append({
            "role": "assistant",
            "content": response_text
        })
else:
    st.info("Upload a PDF to begin.")

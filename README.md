# LLama3-RAG application

scaled down to run faster, kinda stupid now tho.

spinoff of https://github.com/patchy631/ai-engineering-hub/tree/main/document-chat-rag

This project leverages a locally Llama 3 to build a RAG application to **chat with your docs** and Streamlit to build the UI.


## Installation and setup

**Setup Ollama**:
   ```bash
   # pull llama 3, 8B
   ollama pull llama3:8b
   ```
**Setup Qdrant VectorDB**
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
   -v $(pwd)/qdrant_storage:/qdrant/storage:z \
   qdrant/qdrant
   ```

**Install Dependencies**:
   Ensure you have Python 3.11 or later installed.
   ```bash
   pip install streamlit ollama llama-index-vector-stores-qdrant
   ```

**Run app**
   ```bash
   # one terminal
   ollama serve
   # another terminal
   streamlit run app.py
   ```
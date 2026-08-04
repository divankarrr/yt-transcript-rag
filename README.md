# YouTube Transcript RAG Pipeline

An open-source Retrieval-Augmented Generation (RAG) system built with LangChain that extracts transcripts from YouTube videos, stores them in a local vector database, and uses open-source LLMs to answer contextual questions accurately.

## 🚀 Features
- **Transcript Extraction**: Fetches automatic or manual English subtitles directly via video IDs.
- **Semantic Text Splitting**: Chunks dense transcripts into organized windows using `RecursiveCharacterTextSplitter`.
- **Vector Storage**: Employs local `FAISS` structures with `all-MiniLM-L6-v2` embeddings for fast semantic searching.
- **Hallucination Guardrails**: Prompts the `Llama-3.1-8B-Instruct` model to strictly reply with "I don't know" if the text domain does not contain the answer.

## 🛠️ Tech Stack
- **Framework**: LangChain
- **Vector DB**: FAISS
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **LLM**: Meta Llama 3.1 8B Instruct (via HuggingFace Endpoint)

## 📋 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd yt-transcript-rag
   ```

2. **Install required packages:**
   ```bash
   pip install youtube-transcript-api langchain langchain-text-splitters langchain-community faiss-cpu langchain-huggingface python-dotenv
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Hugging Face Access Token:
   ```text
   HUGGINGFACEHAUB_API_TOKEN=your_huggingface_api_token_here
   ```

## 💻 Usage
Run the main script to fetch context from the target video ID and get answers to your questions:
```bash
python app.py
```

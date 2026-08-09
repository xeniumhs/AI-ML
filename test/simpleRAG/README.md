# 🔍 RAG Pipeline — llama3.2 + nomic-embed-text + ChromaDB

A minimal, fully local RAG system. No API keys. No cost. Runs entirely on your machine.

---

## 📁 Project Structure

```
rag_project/
├── docs/                  ← Put your .txt documents here
│   ├── ai_basics.txt
│   └── rag_explained.txt
├── chroma_db/             ← Auto-created: vector store on disk
├── rag_pipeline.py        ← Main script: index docs + run sample queries
├── chat.py                ← Interactive chat mode
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Pull Ollama models (one-time)
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Activate your venv & install dependencies
```bash
conda activate llm        # or: source llm/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Run

### Index documents + sample questions
```bash
python rag_pipeline.py
```

### Interactive chat mode
```bash
python chat.py
```

---

## 🧠 How It Works

```
Your .txt files
      │
      ▼
[TextLoader] → raw text
      │
      ▼
[RecursiveCharacterTextSplitter] → chunks (500 chars, 50 overlap)
      │
      ▼
[OllamaEmbeddings: nomic-embed-text] → vectors
      │
      ▼
[ChromaDB] → stored on disk
      │
   (query time)
      │
      ▼
Your Question → embedded → similarity search → top 3 chunks
      │
      ▼
[llama3.2] receives: system prompt + chunks + question
      │
      ▼
Answer (grounded in your documents)
```

---

## ➕ Add Your Own Documents

1. Drop any `.txt` file into the `docs/` folder
2. Delete the `chroma_db/` folder (to force re-indexing)
3. Run `python rag_pipeline.py` again

---

## 🔧 Config (top of rag_pipeline.py)

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `TOP_K` | 3 | Chunks retrieved per query |
| `LLM_MODEL` | llama3.2 | Ollama LLM model |
| `EMBED_MODEL` | nomic-embed-text | Ollama embedding model |

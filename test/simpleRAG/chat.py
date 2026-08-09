"""
Interactive RAG Chat — type questions, get answers from your docs.
Run this AFTER running rag_pipeline.py at least once (to build the DB).
"""

import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "llama3.2"
TOP_K       = 3


def main():
    if not os.path.exists(CHROMA_DIR):
        print("❌ ChromaDB not found. Run rag_pipeline.py first to index your documents.")
        return

    print("📦 Loading vector store ...")
    embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    llm = OllamaLLM(model=LLM_MODEL, temperature=0)

    prompt_template = """You are a helpful assistant. Use the following context to answer
the question. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    prompt    = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    chain     = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("\n🤖 RAG Chat ready! Type your question (or 'quit' to exit)\n")
    print("=" * 60)

    while True:
        question = input("\n❓ You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break
        if not question:
            continue

        result = chain.invoke({"query": question})
        print(f"\n💬 Answer: {result['result']}")

        sources = set(Path(d.metadata.get("source","?")).name for d in result["source_documents"])
        print(f"📎 Sources: {', '.join(sources)}")


if __name__ == "__main__":
    main()

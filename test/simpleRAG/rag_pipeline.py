import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# =============================================================================
# CONFIG
# =============================================================================

DOCS_DIR = "./simpleRAG/docs"
CHROMA_DIR = "./chroma_db"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

REBUILD_DB = False  # Set True if you add/change documents

# =============================================================================
# LOAD DOCUMENTS
# =============================================================================

def load_documents():

    print(f"\nLoading documents from {DOCS_DIR}")

    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    docs = loader.load()

    print(f"Loaded {len(docs)} document(s)\n")

    # Keep only filename as source
    for doc in docs:
        src = doc.metadata.get("source", "")
        doc.metadata["source"] = Path(src).name

    print("Metadata Example:")
    print(docs[0].metadata)

    return docs


# =============================================================================
# SPLIT DOCUMENTS
# =============================================================================

def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(docs)

    print(f"\nCreated {len(chunks)} chunks")

    print("\nChunk Metadata Example:")
    print(chunks[0].metadata)

    return chunks


# =============================================================================
# BUILD VECTOR DATABASE
# =============================================================================

def build_vectorstore(chunks):

    print("\nCreating embeddings...")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print("Vector database created.\n")

    return vectorstore


# =============================================================================
# LOAD VECTOR DATABASE
# =============================================================================

def load_vectorstore():

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    print("Loaded existing Chroma database.\n")

    return vectorstore


# =============================================================================
# BUILD RAG CHAIN
# =============================================================================

def build_chain(vectorstore):

    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    prompt = PromptTemplate.from_template(
        """
You are a helpful AI assistant.

Use ONLY the supplied context.

If the answer is not present in the context,
say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# =============================================================================
# ASK QUESTION
# =============================================================================

def ask(chain, retriever, question):

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    docs = retriever.invoke(question)

    print("\nRetrieved Chunks")
    print("-" * 70)

    for i, doc in enumerate(docs, 1):

        source = doc.metadata.get("source", "unknown")

        print(f"\nChunk {i}")
        print(f"Source : {source}")
        print(doc.page_content[:250].replace("\n", " "))
        print()

    answer = chain.invoke(question)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\nSources Used")
    print("-" * 70)

    seen = set()

    for doc in docs:

        src = doc.metadata.get("source", "unknown")

        if src not in seen:
            print("•", src)
            seen.add(src)


# =============================================================================
# MAIN
# =============================================================================

def main():

    if REBUILD_DB and os.path.exists(CHROMA_DIR):
        print("Deleting old Chroma database...")
        shutil.rmtree(CHROMA_DIR)

    if not os.path.exists(CHROMA_DIR):

        docs = load_documents()

        chunks = split_documents(docs)

        vectorstore = build_vectorstore(chunks)

    else:

        vectorstore = load_vectorstore()

    chain, retriever = build_chain(vectorstore)

    # ask(chain, retriever, "What is RAG?")
    # ask(chain, retriever, "What are the advantages of RAG?")
    # ask(chain, retriever, "What is machine learning?")
    # ask(chain, retriever, "What is a vector database?")

    while True:

        q = input("\nAsk a question (or type exit): ")

        if q.lower() in ("exit", "quit"):
            break

        ask(chain, retriever, q)


if __name__ == "__main__":
    main()
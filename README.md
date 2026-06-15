# Medical PDF Question Answering System using RAG

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline for answering questions from medical PDF documents.

## Features

* PDF document ingestion
* Text chunking and preprocessing
* Embedding generation using Sentence Transformers
* Semantic search using FAISS
* Question answering using Llama 3 through Ollama

## Tech Stack

* Python
* LangChain
* FAISS
* Sentence Transformers
* Ollama
* Llama 3

## Workflow

PDF → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer Generation

## Example Questions

* What are the symptoms of cardiac arrest?
* What causes cardiac arrest?
* What are the emergency treatment steps?

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Create vector database:
   python ingestion.py

3. Start chatbot:
   python rag_chat.py

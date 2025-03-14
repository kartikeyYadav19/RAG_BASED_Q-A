# RAG-Based Q&A System

This is a Python-based Retrieval-Augmented Generation (RAG) application that:
1. Extracts text from a PDF document.
2. Splits the text into smaller chunks.
3. Generates embeddings and stores them in MongoDB.
4. Retrieves relevant chunks based on a user’s question.
5. Generates an answer using Google’s Gemini API.

## Requirements
- Python 
- Libraries listed in `requirements.txt`

## How to Run
1. Install dependencies:
   ```terminal
   pip install -r requirements.txt
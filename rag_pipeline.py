
import fitz  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from pymongo import MongoClient
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv 

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


MONGO_URI = os.getenv("MONGO_URI") 
client = MongoClient(MONGO_URI)
db = client["rag_database"]  
collection = db["chunks"]   

#Load Sentence Transformers model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#PDF Processing & Chunking
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

#Embedding & Storing in MongoDB Atlas
def get_embedding(text):
    """Get embedding for a text chunk using Sentence Transformers."""
    return embedding_model.encode(text)

def store_chunks_in_mongodb(chunks):
    """Store text chunks and embeddings in MongoDB Atlas."""
    for chunk in chunks:
        embedding = get_embedding(chunk)
        document = {
            "text": chunk,
            "embedding": embedding.tolist()  
        }
        collection.insert_one(document)

#Vector Search
def retrieve_relevant_chunks(question, top_k=3):
    """Retrieve top-k relevant chunks based on cosine similarity."""
    question_embedding = get_embedding(question)
    relevant_chunks = []
    
    # Fetching chunks
    for doc in collection.find():
        chunk_embedding = np.array(doc["embedding"])
        similarity = cosine_similarity([question_embedding], [chunk_embedding])[0][0]
        relevant_chunks.append((similarity, doc["text"]))
    
    # Sort by similarity 
    relevant_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in relevant_chunks[:top_k]]

#Generating Answers
def generate_answer(question, relevant_chunks):
    """Generate an answer using Gemini's LLM."""
    context = "\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")  
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Unable to generate an answer at this time."

# Initial Setup
def initial_setup(pdf_path):
    """Perform the initial setup: Extract text, chunk it, and store embeddings."""
    print("Performing initial setup...")
    
    #Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    #Chunk the text
    print("Chunking text...")
    chunks = chunk_text(text)
    
    #Generate embeddings and storing in MongoDB
    print("Generating embeddings and storing in MongoDB...")
    store_chunks_in_mongodb(chunks)
    
    print("Initial setup complete!")

#Query Handling
def handle_query(question):
    """Handle a user query: Retrieve relevant chunks and generate an answer."""
    print("Handling query...")
    
    #Retrieve relevant chunks
    print("Retrieving relevant chunks...")
    relevant_chunks = retrieve_relevant_chunks(question)
    
    #Generating  and display the answer
    print("Generating answer...")
    answer = generate_answer(question, relevant_chunks)
    print("\nAnswer:", answer)

#Main Method
def main(pdf_path, question):
    """Main pipeline for the RAG-based Q&A system."""
    
    if not collection.count_documents({}):  #Looking if it is empty
        initial_setup(pdf_path)
    
    
    handle_query(question)

#pipeline
if __name__ == "__main__":
    #PDF path dynamically
    current_directory = os.path.dirname(os.path.abspath(__file__))  
    pdf_folder = "data"  
    pdf_filename = "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"  
    pdf_path = os.path.join(current_directory, pdf_folder, pdf_filename) 
    
    # Question Prompt
    question = "How to cure cancer?"
    
    # Execution
    main(pdf_path, question)
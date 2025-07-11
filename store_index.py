from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

extracted_data = load_pdf("/Users/donghunshin/Documents/End-to-end-Medical-Chatbot-using-Llama2/data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Create FAISS index from documents
vectorstore = FAISS.from_texts([t.page_content for t in text_chunks], embeddings)

# Save the FAISS index locally
vectorstore.save_local("faiss_index")
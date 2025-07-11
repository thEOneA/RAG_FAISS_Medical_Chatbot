from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')

# Disable auto-reload and debugging in production
app.config['TEMPLATES_AUTO_RELOAD'] = False

def initialize_models():
    """Initialize all models and chains."""
    logger.info("Loading embeddings model...")
    embeddings = download_hugging_face_embeddings()

    logger.info("Loading FAISS index...")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    logger.info("Loading LLM model...")
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={'max_new_tokens': 512, 'temperature': 0.8}
    )

    logger.info("Setting up QA chain...")
    prompt = PromptTemplate(template=prompt_template, 
                          input_variables=["context", "question"])
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# Initialize components
load_dotenv()
qa_chain = initialize_models()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        logger.info(f"Received query: {msg}")
        # Use invoke instead of __call__
        result = qa_chain.invoke({"query": msg})
        logger.info("Generated response successfully")
        return str(result["result"])
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return str("Sorry, I encountered an error processing your request.")

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(
        host="0.0.0.0", 
        port=8080,
        debug=False,  # Disable debug mode
        use_reloader=False  # Disable auto-reloader
    )

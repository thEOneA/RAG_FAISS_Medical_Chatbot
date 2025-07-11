from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdf(data):
    """Extract data from PDF files in directory."""
    try:
        loader = DirectoryLoader(data,
                        glob="*.pdf",
                        loader_cls=PyPDFLoader)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        raise

def text_split(extracted_data):
    """Create text chunks from documents."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20
        )
        text_chunks = text_splitter.split_documents(extracted_data)
        logger.info(f"Created {len(text_chunks)} text chunks")
        return text_chunks
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        raise

def download_hugging_face_embeddings():
    """Initialize and return HuggingFace embeddings model."""
    try:
        logger.info("Loading HuggingFace embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="models/embeddings"  # Cache embeddings locally
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise
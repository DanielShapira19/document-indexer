import os
import sys
import logging
import psycopg2
import google.generativeai as genai
from contextlib import contextmanager
from dotenv import load_dotenv
import nltk
from abc import ABC, abstractmethod
from pypdf import PdfReader
from docx import Document

# =============================================================================
# 0. CONFIGURATION & SETUP
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# Configure logging: writes to both console and 'indexing.log' file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("indexing.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Retrieve environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

# Validate that required variables exist
if not GEMINI_API_KEY or not POSTGRES_URL:
    logger.critical("Error: Missing environment variables in .env file.")
    logger.critical("Please ensure GEMINI_API_KEY and POSTGRES_URL are set.")
    sys.exit(1)

# Configure Google Gemini API
genai.configure(api_key=GEMINI_API_KEY)


# =============================================================================
# 1. DATABASE UTILITIES (PostgreSQL)
# =============================================================================

@contextmanager
def get_db_connection():
    """
    Context manager for PostgreSQL database connection.
    Ensures the connection is closed properly even if an error occurs.
    """
    conn = None
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        yield conn
    except psycopg2.DatabaseError as e:
        logger.error(f"[DB Connection Error] Failed to connect: {e}")
        raise
    finally:
        if conn:
            conn.close()


def setup_database():
    """
    Creates the required table in the database if it doesn't exist.
    Uses FLOAT8[] for embeddings to avoid pgvector dependency issues.
    """
    schema_query = """
    CREATE TABLE IF NOT EXISTS document_vectors (
        id SERIAL PRIMARY KEY,
        chunk_text TEXT NOT NULL,
        embedding FLOAT8[],      -- Storing vector as a standard float array
        filename TEXT NOT NULL,
        strategy_split TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_query)
                conn.commit()  # Commit changes to save the table

        logger.info("Database schema ensured successfully (Table 'document_vectors' is ready).")
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False




# =============================================================================
# 2. HELPER FUNCTIONS & NLTK SETUP
# =============================================================================

# Ensure NLTK data is available for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


def db_insert_batch(chunks_data):
    """
    Inserts a batch of processed chunks into the database.
    Input: List of tuples (chunk_text, embedding, filename, strategy_split)
    """
    insert_query = """
    INSERT INTO document_vectors (chunk_text, embedding, filename, strategy_split)
    VALUES (%s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # executemany is optimized for bulk inserts
                cur.executemany(insert_query, chunks_data)
                conn.commit()
        logger.info(f"Successfully saved {len(chunks_data)} chunks to DB.")
    except Exception as e:
        logger.error(f"Insert batch failed: {e}")


# =============================================================================
# 3. ABSTRACT PROCESSOR (BASE CLASS) - UPDATED
# =============================================================================

class BaseDocumentProcessor(ABC):
    def __init__(self, file_path, **kwargs):
        """
        Changed: Now accepts **kwargs to handle dynamic options
        like chunk_size or overlap without changing the signature for everyone.
        """
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.options = kwargs  # Store options for use by child classes

    @property
    @abstractmethod
    def strategy_name(self):
        pass

    def extract_text(self):
        """Reads raw text from PDF or DOCX files."""
        if not os.path.exists(self.file_path):
            logger.error(f"File not found: {self.file_path}")
            return None

        _, ext = os.path.splitext(self.file_path)
        ext = ext.lower()
        text = ""

        try:
            if ext == '.pdf':
                reader = PdfReader(self.file_path)
                for page in reader.pages:
                    extract = page.extract_text()
                    if extract: text += extract + "\n"
            elif ext == '.docx':
                doc = Document(self.file_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            else:
                logger.error(f"Unsupported file format: {ext}")
                return None
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading file {self.filename}: {e}")
            return None

    @abstractmethod
    def split_text(self, text):
        pass

    def generate_embeddings(self, chunks):
        embeddings = []
        model = "models/embedding-001"
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        try:
            for chunk in chunks:
                result = genai.embed_content(
                    model=model,
                    content=chunk,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            return []
        return embeddings

    def run(self):
        logger.info(f"--- Processing: {self.filename} | Strategy: {self.strategy_name} ---")

        text = self.extract_text()
        if not text: return

        chunks = self.split_text(text)
        if not chunks:
            logger.warning("No text chunks created.")
            return

        embeddings = self.generate_embeddings(chunks)
        if len(embeddings) != len(chunks):
            logger.error("Mismatch: Chunks vs Embeddings count.")
            return

        data_to_save = []
        for chunk, vec in zip(chunks, embeddings):
            data_to_save.append((chunk, vec, self.filename, self.strategy_name))

        db_insert_batch(data_to_save)


# =============================================================================
# 4. CONCRETE STRATEGIES (IMPLEMENTATIONS) - UPDATED
# =============================================================================

class FixedSizeProcessor(BaseDocumentProcessor):
    @property
    def strategy_name(self):
        return "fixed"

    def split_text(self, text):
        # Use options: try to get from user input, otherwise use default values
        chunk_size = self.options.get('chunk_size', 500)
        overlap = self.options.get('overlap', 50)

        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 10:
                chunks.append(chunk)
        return chunks


class SentenceProcessor(BaseDocumentProcessor):
    @property
    def strategy_name(self): return "sentence"

    def split_text(self, text): return nltk.sent_tokenize(text)


class ParagraphProcessor(BaseDocumentProcessor):
    @property
    def strategy_name(self): return "paragraph"

    def split_text(self, text):
        raw = text.split('\n\n')
        return [c.strip() for c in raw if c.strip()]










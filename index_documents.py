import os
import sys
import logging
import psycopg2
import google.generativeai as genai
from contextlib import contextmanager
from dotenv import load_dotenv

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

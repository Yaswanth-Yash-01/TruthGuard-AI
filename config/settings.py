import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "truthguard")

EMBEDDING_MODEL: str = "models/text-embedding-004"
# Dimension is discovered at runtime from whichever model the API key can access.
LLM_MODEL: str = "gemini-1.5-flash"
TOP_K: int = 5
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
MAX_PAGES: int = 100

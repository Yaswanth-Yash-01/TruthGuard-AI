import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")   # used for embeddings only
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "truthguard")

EMBEDDING_MODEL: str = "models/text-embedding-004"
LLM_MODEL: str = "command-r-plus"
TOP_K: int = 5
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
MAX_PAGES: int = 100

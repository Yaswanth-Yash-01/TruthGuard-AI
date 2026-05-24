import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "truthguard")

EMBEDDING_MODEL: str = "embed-english-v3.0"
LLM_MODEL: str = "command-r-plus-08-2024"
TOP_K: int = 5
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
MAX_PAGES: int = 100

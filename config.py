from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
EXTRACTED_TEXT_DIR = ROOT_DIR / "extracted_text"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
FAISS_INDEX_DIR = VECTOR_STORE_DIR / "faiss_index"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.pkl"
PROMPT_PATH = ROOT_DIR / "prompts" / "prompt.txt"

DEFAULT_CHUNK_SIZE = 220
DEFAULT_CHUNK_OVERLAP = 40
DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_MODEL = "simple-bow"

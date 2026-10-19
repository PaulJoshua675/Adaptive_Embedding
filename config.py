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
DEFAULT_TOP_K = 10
MIN_RETRIEVAL_SIMILARITY = 0.25
MIN_RERANKER_SCORE = 0.0
DEFAULT_EMBEDDING_MODEL = "simple-bow"

EMBEDDING_MODELS = {
	"bge": "BAAI/bge-small-en-v1.5",
	"e5": "intfloat/e5-small-v2",
	"minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DOMAIN_EXAMPLE_QUESTIONS = {
    "cs.CV": (
        "How does the CNN in Automatic Image Colourizer predict colours from grayscale images (2312.09876)?",
        "What training-data challenges does Automatic Image Colourizer identify for CNN-based colour prediction (2312.09876)?",
    ),
    "cs.CL": (
        "How often were InstructGPT PPO-ptx outputs preferred to the 175B SFT model in 2203.02155?",
        "How does the 1.3B PPO-ptx model compare with GPT-3 in human evaluations in 2203.02155?",
    ),
    "cs.LG": (
        "How do the per-token KL penalty in PPO and the KL reward coefficient affect alignment and DROP/SQuAD performance in 2203.02155?",
        "How does PPO-ptx mix pretraining gradients with PPO updates to reduce the alignment tax in 2203.02155?",
    ),
    "cs.AI": (
        "What explains why retrieved passages are not read by search-augmented language models (2610.21731)?",
        "What does the multi-agent LLM system in 2610.13663 do to tune specialized neural-network models?",
    ),
    "cs.IR": (
        "How does AiSAQ-in-Filtering support DRAM-free filtered information retrieval (2610.17627)?",
        "What factors explain why retrieved passages are not read by search-augmented language models (2610.21731)?",
    ),
}

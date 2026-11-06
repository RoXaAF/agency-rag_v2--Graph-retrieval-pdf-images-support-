"""All settings in one place."""
from pathlib import Path

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "chunks.db"
GRAPH_PATH = DATA_DIR / "graph.json"
IMAGES_DIR = DATA_DIR / "images"
CACHE_DIR = DATA_DIR / "cache"

LLM_MODEL = "gemma4:12b"      # exact tag from `ollama list`
NUM_CTX = 8192
KEEP_ALIVE = "30m"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_IMAGE_PX = 100

PROMPT_VERSION = "v1"         # bump to invalidate the triple cache
FUZZY_MERGE = 90              # rapidfuzz score for merging entities at build time
FUZZY_LINK = 85               # rapidfuzz score for linking question entities
HOPS = 2
MAX_FACTS = 20
MAX_CHUNKS = 6
MAX_EDGES_PER_NODE = 25       # caps expansion through very common nodes

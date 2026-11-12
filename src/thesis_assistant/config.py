"""Central, environment-driven configuration.

Every tunable value lives here so that ingestion and query time cannot drift
apart (for example by using different embedding models). Values are read once
at import time from the process environment, which ``load_dotenv`` populates
from a local ``.env`` file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repository root (this file is src/thesis_assistant/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Providers ------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")
CHAT_MODEL_PROVIDER = os.getenv("CHAT_MODEL_PROVIDER", "deepseek")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Embeddings and vector store ---------------------------------------------------

# The embedding model and index must be identical at ingestion and query time.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# Pinecone index names allow only lowercase letters, digits and hyphens.
INDEX_NAME = os.getenv("INDEX_NAME_ASOIU_02", "rag-asoiu-openai-v2")

# --- Retrieval and prompt limits -------------------------------------------------------

RETRIEVAL_TOP_K = 3  # chunks returned by the retrieval tool
# "dense": vector similarity only. "hybrid": over-fetch RETRIEVAL_FETCH_K
# candidates, then fuse the dense order with a BM25 lexical ranking.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense").lower()
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "12"))
MAX_CHARS_PER_CHUNK = 2000  # characters of each chunk sent to the model
MAX_HISTORY_TURNS = 3  # prior user/assistant exchanges included in the prompt
MAX_DOCUMENT_CHARS = 30_000  # uploaded-document characters included in the prompt

# --- Vision fallback ---------------------------------------------------------------------

VISION_FALLBACK_ENABLED = os.getenv("VISION_FALLBACK_ENABLED", "true").lower() == "true"
VISION_MODEL = os.getenv("VISION_MODEL", "openai/gpt-4o-mini")
VISION_FALLBACK_MIN_CHARS = int(os.getenv("VISION_FALLBACK_MIN_CHARS", "40"))
VISION_FALLBACK_DPI = int(os.getenv("VISION_FALLBACK_DPI", "150"))
MAX_VISION_OUTPUT_TOKENS = 1000

# --- Ingestion -------------------------------------------------------------------------------

# Corpus root: one sub-folder per document category.
DB_ROOT = Path(os.getenv("ASOIU_DB_ROOT", PROJECT_ROOT / "DB_ASOIU"))
MANIFEST_PATH = Path(os.getenv("INGESTION_MANIFEST_PATH", PROJECT_ROOT / "ingestion_manifest.json"))
WEB_URLS = ["https://asoiu.edu.az/"]
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
UPSERT_BATCH_SIZE = 25
BATCH_PAUSE_SECONDS = 1  # pause between batches to respect provider rate limits

# --- Observability -----------------------------------------------------------------------------

LOG_FORMAT = os.getenv("LOG_FORMAT", "json").lower()  # "json" or "text"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
METRICS_PORT = int(os.getenv("METRICS_PORT", "0"))  # 0 disables the metrics server

# --- Web app ---------------------------------------------------------------------------------

# Per-user accounts: "name:salt$hash;name2:salt$hash" (generate entries with
# `python -m thesis_assistant.auth`). APP_PASSWORD is a legacy shared-password
# fallback. If neither is set the web app is open.
APP_USERS = os.getenv("APP_USERS", "")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# --- Telegram bot ----------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# Numeric user IDs (usernames can change). Empty means open to everyone.
TELEGRAM_ALLOWED_USER_IDS = {
    int(uid)
    for uid in os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").replace(" ", "").split(",")
    if uid
}

# --- State, rate limiting and usage accounting ---------------------------------------------------

# SQLite file holding Telegram chat history and the usage log.
STATE_DB_PATH = Path(os.getenv("STATE_DB_PATH", PROJECT_ROOT / "state" / "thesis_assistant.db"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

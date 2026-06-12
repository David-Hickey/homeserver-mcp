import os
from pathlib import Path


class Environment:
    def __init__(self):
        self.memos_db = Path(os.getenv("MEMOS_DB", "/external-data/memos/memos.db"))
        self.rag_db = Path(os.getenv("MEMOS_RAG_DB", "/mcp/data/memos/rag.db"))
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.embed_model = os.getenv("MEMOS_EMBED_MODEL", "nomic-embed-text")
        self.embed_dim = int(os.getenv("MEMOS_EMBED_DIM", "768"))
        self.batch_size = int(os.getenv("MEMOS_BATCH_SIZE", "32"))
        self.rewrite_model = os.getenv("RAG_REWRITE_MODEL", "qwen2.5:3b")

import requests
import struct

def encode_vector(floats: list[float]) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)

def decode_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))

def embed_texts(texts: list[str], ollama_url: str, embed_model: str) -> list[list[float]]:
    """Call Ollama's /api/embed endpoint for a batch of texts."""
    resp = requests.post(
        f"{ollama_url}/api/embed",
        json={"model": embed_model, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

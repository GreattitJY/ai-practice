"""스크립트들이 함께 쓰는 경로, 설정, 헬퍼."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RAW_PRODUCTS = DATA / "raw_products.json"
PRODUCTS_CSV = DATA / "products.csv"
PRODUCTS_FLAT = DATA / "products_flat.jsonl"
EMBEDDINGS = DATA / "embeddings.npy"
EMBEDDINGS_META = DATA / "embeddings_meta.json"
USERS = DATA / "users.json"

LMSTUDIO_URL = "http://localhost:1234/v1/embeddings"
MODEL = "text-embedding-qwen3-embedding-4b"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """벡터(또는 행렬의 각 행)의 길이를 1로 맞춘다. 코사인 유사도 = 내적이 되게 하려는 것."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

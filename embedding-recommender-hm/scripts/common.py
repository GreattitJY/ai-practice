"""스크립트들이 함께 쓰는 경로, 설정, 헬퍼."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"

RAW_ARTICLES = RAW / "articles.csv"
RAW_TRANSACTIONS = RAW / "transactions.parquet"  # transactions_train.csv를 parquet로 바꾼 것

# 01_build_table.py 결과
PRODUCTS = DATA / "products.parquet"          # 상품(product_code) 단위 테이블, 판매량 많은 순
INTERACTIONS = DATA / "interactions.parquet"  # (날짜, 고객 번호, 상품 번호) — 문자열 id를 정수로 바꿔 메모리를 줄임
CUSTOMERS = DATA / "customers.json"           # 고객 번호 → 원래 customer_id

# 02_flatten.py 결과
PRODUCTS_FLAT = DATA / "products_flat.jsonl"

# 03_embed.py 결과
EMBEDDINGS = DATA / "embeddings.f32"          # (상품 수, 차원) float32 memmap. 앞에서부터 채워진다
EMBED_PROGRESS = DATA / "embed_progress.json"

LMSTUDIO_URL = "http://localhost:1234/v1/embeddings"
MODEL = "text-embedding-qwen3-embedding-4b"
DIM = 2560


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """벡터(또는 행렬의 각 행)의 길이를 1로 맞춘다. 코사인 유사도 = 내적이 되게 하려는 것."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embedded_count() -> int:
    """지금까지 임베딩이 끝난 상품 수. 판매량 순위 0 ~ (n-1)번까지 벡터가 있다."""
    if not EMBED_PROGRESS.exists():
        return 0
    return json.loads(EMBED_PROGRESS.read_text())["done"]


def load_embeddings(dim: int | None = None) -> np.ndarray:
    """임베딩이 끝난 상품의 벡터만 읽는다. dim을 주면 MRL 절단 후 재정규화(3-2강 p8)."""
    n = embedded_count()
    if n == 0:
        raise SystemExit("임베딩된 상품이 없습니다. scripts/03_embed.py를 먼저 실행하세요.")
    total = json.loads(EMBED_PROGRESS.read_text())["total"]
    matrix = np.array(np.memmap(EMBEDDINGS, dtype=np.float32, mode="r", shape=(total, DIM))[:n])
    if dim:
        matrix = l2_normalize(matrix[:, :dim])
    return matrix

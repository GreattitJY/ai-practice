"""3단계: 평탄화한 문장을 벡터로 변환.

모델: Qwen3-Embedding-4B (LM Studio 로컬 서버, 2560차원, MRL 지원)
- 이 실습은 상품 ↔ 상품을 비교하는 대칭 구조라 검색용 `Instruct:` 접두어를 붙이지 않는다.
  (3-2강 p5: 지시문은 질의에만 붙는 검색용 형식. 양쪽이 같은 종류면 같은 형식으로 둔다.)
- 받은 벡터는 L2 정규화해서 저장한다(3-2강 p15: last-token 풀링 + L2 정규화).

사전 준비: `lms server start` 후 `lms load text-embedding-qwen3-embedding-4b`
"""

import json
import urllib.error
import urllib.request

import numpy as np

from common import EMBEDDINGS, EMBEDDINGS_META, LMSTUDIO_URL, MODEL, PRODUCTS_FLAT, l2_normalize, read_jsonl

BATCH_SIZE = 16


def embed(texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    req = urllib.request.Request(LMSTUDIO_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as res:
            data = json.load(res)["data"]
    except urllib.error.URLError as e:
        raise SystemExit(f"LM Studio 서버에 연결할 수 없습니다({e}). `lms server start`를 실행하세요.")
    return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


def main() -> None:
    records = read_jsonl(PRODUCTS_FLAT)
    vectors = []
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        vectors.extend(embed([r["text"] for r in batch]))
        print(f"  {min(i + BATCH_SIZE, len(records))}/{len(records)}")
    matrix = l2_normalize(np.array(vectors, dtype=np.float32))
    np.save(EMBEDDINGS, matrix)
    EMBEDDINGS_META.write_text(
        json.dumps({"model": MODEL, "dim": matrix.shape[1], "ids": [r["id"] for r in records]}, indent=2),
        encoding="utf-8",
    )
    print(f"{matrix.shape[0]}개 × {matrix.shape[1]}차원 → {EMBEDDINGS.name}")


if __name__ == "__main__":
    main()

"""3단계: 평탄화한 문장을 벡터로 변환 — 중간 저장과 이어받기 지원.

모델: Qwen3-Embedding-4B (LM Studio 로컬 서버, 2560차원, MRL 지원)
- 상품 ↔ 상품을 비교하는 대칭 구조라 검색용 `Instruct:` 접두어를 붙이지 않는다(3-2강 p5).
- 받은 벡터는 L2 정규화해서 저장한다(3-2강 p15).

판매량 많은 순서대로 앞에서부터 채운다. 배치 하나가 끝날 때마다 진행 상황을 저장하므로
중간에 멈춰도(Ctrl+C, 잠자기, 오류) 다시 실행하면 멈춘 곳부터 이어서 한다.

사용 예
  caffeinate -i .venv/bin/python scripts/03_embed.py --limit 100     # 시험
  caffeinate -i .venv/bin/python scripts/03_embed.py --limit 10000   # 많이 팔린 1만 개까지
  caffeinate -i .venv/bin/python scripts/03_embed.py                 # 전체
  (caffeinate -i: 실행하는 동안 맥이 잠자기에 들어가지 않게 한다)

사전 준비: `lms server start` 후 `lms load text-embedding-qwen3-embedding-4b`
"""

import argparse
import json
import time
import urllib.error
import urllib.request

import numpy as np

from common import DIM, EMBED_PROGRESS, EMBEDDINGS, LMSTUDIO_URL, MODEL, PRODUCTS_FLAT, embedded_count, l2_normalize, read_jsonl


def embed(texts: list[str]) -> np.ndarray:
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    req = urllib.request.Request(LMSTUDIO_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            data = json.load(res)["data"]
    except urllib.error.URLError as e:
        raise SystemExit(f"LM Studio 서버에 연결할 수 없습니다({e}). `lms server start`를 실행하세요.")
    vectors = np.array([d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float32)
    if vectors.shape[1] != DIM:
        raise SystemExit(f"차원이 {vectors.shape[1]}입니다. {MODEL}({DIM}차원)이 로드됐는지 확인하세요.")
    return l2_normalize(vectors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="판매량 상위 몇 개까지 임베딩할지(기본: 전체)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    records = read_jsonl(PRODUCTS_FLAT)
    total = len(records)
    target = min(args.limit or total, total)
    start = embedded_count()
    if start >= target:
        print(f"이미 {start:,}개까지 끝났습니다(목표 {target:,}).")
        return

    mode = "r+" if EMBEDDINGS.exists() else "w+"
    matrix = np.memmap(EMBEDDINGS, dtype=np.float32, mode=mode, shape=(total, DIM))
    print(f"{start:,}번부터 {target:,}번까지 임베딩 (전체 {total:,}개)")

    began = time.time()
    for i in range(start, target, args.batch_size):
        batch = records[i : min(i + args.batch_size, target)]
        matrix[i : i + len(batch)] = embed([r["text"] for r in batch])
        matrix.flush()
        done = i + len(batch)
        EMBED_PROGRESS.write_text(json.dumps({"done": done, "total": total, "model": MODEL, "dim": DIM}))
        rate = (done - start) / (time.time() - began)
        eta = (target - done) / rate / 60
        print(f"  {done:,}/{target:,}  {rate:.1f}건/초  남은 시간 약 {eta:.0f}분", flush=True)
    print("완료")


if __name__ == "__main__":
    main()

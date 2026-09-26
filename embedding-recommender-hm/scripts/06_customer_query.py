"""고객을 문장으로 바꿔 질의로 쓰기 — 지시문(Instruct) 실습.

지금까지(04·05)는 상품 ↔ 상품을 비교해서 지시문을 붙이지 않았다.
여기서는 "고객 문장 → 상품"으로 질의와 문서의 종류가 달라서,
Qwen3-Embedding 형식대로 질의(고객 문장)에만 지시문을 붙인다(3-2강 p5, p15, p18).
상품 벡터(03_embed.py 결과)는 그대로 다시 쓴다.

1. 고객 문장: 학습 기간(정답 기간 이전)의 정보만으로 만든다. 정답 기간의 구매가 섞이면 안 된다.
   - 나이(customers.csv)
   - 최근 구매 20개의 대상 코너(index_name)·종류(product_type) 집계, 가장 최근 구매 5개의 이름
   - 넣지 않는 것: 멤버십 상태·뉴스 수신 여부. 취향이 아니라 마케팅 수신 설정이다.
2. 비교할 변형
   - instruct:       지시문 + 나이 + 구매 요약
   - plain:          나이 + 구매 요약 (지시문 없음)
   - instruct_noage: 지시문 + 구매 요약 (나이 없음)
3. 평가: 05_evaluate.py와 같은 고객·같은 정답으로 인기순, 평균 벡터 방식과 비교한다.

고객 벡터는 data/customer_query/<변형>.npy에 저장해 두고, 있으면 다시 임베딩하지 않는다.
LM Studio는 저장된 벡터가 없는 변형을 임베딩할 때만 필요하다(고객 2,000명에 약 9분).

사용 예
  caffeinate -i .venv/bin/python scripts/06_customer_query.py
  .venv/bin/python scripts/06_customer_query.py --show 3     # 고객 문장 예시만 보기
"""

import argparse
import importlib
import json
from collections import Counter

import numpy as np
import pandas as pd

from common import CUSTOMERS, DATA, PRODUCTS, RAW, load_embeddings

evaluate = importlib.import_module("05_evaluate")
recommender = evaluate.recommender
embedder = importlib.import_module("03_embed")

OUT = DATA / "customer_query"
TASK = "Given an H&M customer's profile and recent purchases, retrieve fashion products the customer is likely to buy next"
VARIANTS = {
    "instruct": {"instruct": True, "age": True},
    "plain": {"instruct": False, "age": True},
    "instruct_noage": {"instruct": True, "age": False},
}
PROFILE_HISTORY = 20
RECENT_NAMES = 5


def top_counts(values, n: int) -> str:
    return ", ".join(f"{v.lower()} ({c})" for v, c in Counter(values).most_common(n))


def profile(age: float | None, recent: list[int], products: pd.DataFrame) -> str:
    """recent: 학습 기간 구매 상품 번호, 최근 순, 중복 없음."""
    p = products.iloc[recent]
    who = f"H&M customer, age {int(age)}." if age is not None else "H&M customer."
    names = "; ".join(f"{r.prod_name} ({r.product_type.lower()})" for r in p.head(RECENT_NAMES).itertuples())
    return (
        f"{who} Recent purchases mostly from {top_counts(p.index_name, 2)}. "
        f"Often buys {top_counts(p.product_type, 3)}. "
        f"Latest items: {names}."
    )


def build_profiles(e: dict, variant: dict) -> list[str]:
    products = pd.read_parquet(PRODUCTS)
    ids = json.loads(CUSTOMERS.read_text())
    ages = pd.read_csv(RAW / "customers.csv", usecols=["customer_id", "age"]).set_index("customer_id").age

    train = e["train_all"]
    wanted = set(e["customers"])
    recent = (
        train[train.customer_idx.isin(wanted)]
        .sort_values("t_dat", ascending=False, kind="stable")
        .drop_duplicates(["customer_idx", "product_idx"])
        .groupby("customer_idx").product_idx.agg(lambda s: s.head(PROFILE_HISTORY).tolist())
    )
    texts = []
    for c in e["customers"]:
        age = ages.get(ids[c]) if variant["age"] else None
        text = profile(None if pd.isna(age) else age, recent[c], products)
        texts.append(f"Instruct: {TASK}\nQuery: {text}" if variant["instruct"] else text)
    return texts


def customer_vectors(name: str, e: dict) -> np.ndarray:
    path = OUT / f"{name}.npy"
    if path.exists():
        return np.load(path)
    texts = build_profiles(e, VARIANTS[name])
    print(f"[{name}] 고객 {len(texts):,}명 임베딩")
    vectors = []
    for i in range(0, len(texts), 64):
        vectors.append(embedder.embed(texts[i : i + 64]))
        print(f"  {min(i + 64, len(texts)):,}/{len(texts):,}", flush=True)
    matrix = np.concatenate(vectors)
    OUT.mkdir(exist_ok=True)
    np.save(path, matrix)
    (OUT / f"{name}.jsonl").write_text("".join(json.dumps({"customer_idx": int(c), "text": t}, ensure_ascii=False) + "\n" for c, t in zip(e["customers"], texts)))
    return matrix


def recommend_by_query(queries: np.ndarray, matrix: np.ndarray, hist: list[list[int]], k: int) -> np.ndarray:
    scores = queries @ matrix.T
    for i, h in enumerate(hist):
        scores[i, h] = -np.inf
    return np.argsort(-scores, axis=1)[:, :k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=12)
    parser.add_argument("--customers", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", default=",".join(VARIANTS), help="쉼표로 구분")
    parser.add_argument("--show", type=int, help="고객 문장 예시 n개만 출력하고 끝냄")
    args = parser.parse_args()

    e = evaluate.eval_set(args.customers, 20, args.seed)
    # 고객 문장은 임베딩되지 않은 상품까지 포함한 학습 기간 전체 구매로 만든다
    from common import INTERACTIONS
    interactions = pd.read_parquet(INTERACTIONS)
    e["train_all"] = interactions[interactions.t_dat <= e["cutoff"]]

    if args.show:
        e["customers"] = e["customers"][: args.show]
        for text in build_profiles(e, VARIANTS["instruct"]):
            print(text, "\n")
        return

    matrix = load_embeddings()
    hist, ans, k = e["hist"], e["ans"], args.k
    rows = {
        "인기순 (직전 7일)": evaluate.score(evaluate.popularity_recommend(evaluate.popular_before(e["train"], e["cutoff"]), hist, k), ans, k),
        "평균 벡터 (05와 동일)": evaluate.score(recommender.recommend(matrix, hist, k), ans, k),
    }
    for name in args.variants.split(","):
        rows[f"고객 문장: {name}"] = evaluate.score(recommend_by_query(customer_vectors(name, e), matrix, hist, k), ans, k)
    print(f"\n평가 고객 {len(hist):,}명, 상품 {len(matrix):,}개\n")
    evaluate.print_table(rows, k)


if __name__ == "__main__":
    main()

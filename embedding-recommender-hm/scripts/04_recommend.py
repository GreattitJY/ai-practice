"""4~5단계: 고객의 구매 목록 평균 벡터로 Top-k 추천.

1. 고객이 산 상품들(최근 --history개)의 벡터를 평균 낸다.
2. 평균 벡터를 다시 L2 정규화한다. 평균을 내면 길이가 1보다 짧아지기 때문(3-2강 p8).
3. 임베딩된 모든 상품과 코사인 유사도(= 정규화된 벡터끼리의 내적)를 구한다.
4. 이미 산 상품을 빼고 점수가 높은 k개를 추천한다.

임베딩이 끝난 상품(판매량 상위 N개)만 대상으로 한다. 구매 목록 중 임베딩되지 않은 상품은 건너뛴다.

사용 예
  python scripts/04_recommend.py                   # 무작위 고객 3명
  python scripts/04_recommend.py --customer 12345  # 고객 번호(customer_idx) 지정
  python scripts/04_recommend.py --product 0       # 상품 하나와 비슷한 상품
  python scripts/04_recommend.py --dim 256         # MRL: 앞 256차원만 사용
"""

import argparse

import numpy as np
import pandas as pd

from common import INTERACTIONS, PRODUCTS, l2_normalize, load_embeddings


def user_vectors(matrix: np.ndarray, histories: list[list[int]]) -> np.ndarray:
    return l2_normalize(np.stack([matrix[h].mean(axis=0) for h in histories]))


def recommend(matrix: np.ndarray, histories: list[list[int]], k: int) -> np.ndarray:
    """histories[i]의 평균 벡터로 고객 i에게 k개 추천. 결과는 (고객 수, k) 상품 번호."""
    scores = user_vectors(matrix, histories) @ matrix.T
    for i, h in enumerate(histories):
        scores[i, h] = -np.inf
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, axis=1)


def recent_history(purchases: pd.DataFrame, n_embedded: int, size: int) -> pd.Series:
    """고객별로 임베딩된 상품만 남겨, 최근에 산 순서로 중복 없이 size개."""
    p = purchases[purchases.product_idx < n_embedded].sort_values("t_dat", ascending=False, kind="stable")
    p = p.drop_duplicates(["customer_idx", "product_idx"])
    return p.groupby("customer_idx").product_idx.agg(lambda s: s.head(size).tolist())


def describe(products: pd.DataFrame, idx: int) -> str:
    p = products.iloc[idx]
    return f"[{idx:>5}] {p.prod_name} ({p.product_type}, {p.index_name})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer", type=int, action="append", help="고객 번호. 여러 번 줄 수 있음")
    parser.add_argument("--product", type=int, help="이 상품과 비슷한 상품 보기")
    parser.add_argument("--history", type=int, default=20, help="최근 구매 몇 개로 평균을 낼지")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--dim", type=int, help="MRL 절단 차원(예: 256)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matrix = load_embeddings(args.dim)
    products = pd.read_parquet(PRODUCTS)
    print(f"임베딩된 상품 {len(matrix):,}개 × {matrix.shape[1]}차원, Top-{args.k}")

    if args.product is not None:
        print(f"\n■ {describe(products, args.product)}와 비슷한 상품")
        for rank, idx in enumerate(recommend(matrix, [[args.product]], args.k)[0], 1):
            print(f"  {rank:>2}. {float(matrix[idx] @ matrix[args.product]):.4f}  {describe(products, idx)}")
        return

    interactions = pd.read_parquet(INTERACTIONS)
    histories = recent_history(interactions, len(matrix), args.history)
    customers = args.customer or histories.sample(3, random_state=args.seed).index.tolist()
    missing = [c for c in customers if c not in histories.index]
    if missing:
        raise SystemExit(f"임베딩된 상품을 산 적이 없는 고객: {missing}")

    recs = recommend(matrix, [histories[c] for c in customers], args.k)
    for c, rec in zip(customers, recs):
        print(f"\n■ 고객 {c} — 최근 구매 {len(histories[c])}개")
        for idx in histories[c][:5]:
            print("    산 것:", describe(products, idx))
        if len(histories[c]) > 5:
            print(f"    … 외 {len(histories[c]) - 5}개")
        for rank, idx in enumerate(rec, 1):
            print(f"  {rank:>2}. {describe(products, idx)}")


if __name__ == "__main__":
    main()

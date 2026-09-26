"""평가: 실제 구매 기록으로 "다음 주에 살 상품을 맞혔는가"를 잰다.

1. 기간 나누기: 마지막 7일(2020-09-16 ~ 09-22)을 정답으로 떼어 두고, 그 전 구매로 추천을 만든다.
2. 대상 고객: 두 기간 모두 임베딩된 상품을 산 고객 중 --customers명을 무작위로 뽑는다.
3. 정답: 마지막 7일에 산 상품 중 전에 산 적 없는 상품.
   추천에서 이미 산 상품을 빼므로, 다시 산 상품은 정답에서도 뺀다.
4. 점수(Top-k 기준)
   - HitRate: 정답을 하나라도 맞힌 고객 비율
   - Recall: 고객별로 정답 중 몇 %를 맞혔는지의 평균
   - Precision: 추천 k개 중 정답의 비율 평균
5. 비교 대상
   - 인기순: 정답 기간 직전 7일에 가장 많이 팔린 상품(이미 산 것 제외). 임베딩 없이 만드는 가장 단순한 추천
   - 임베딩: 전체 차원과 MRL 절단 차원(256·64)

사용 예
  python scripts/05_evaluate.py
  python scripts/05_evaluate.py -k 20 --customers 5000
"""

import argparse
import importlib

import numpy as np
import pandas as pd

from common import INTERACTIONS, load_embeddings

recommender = importlib.import_module("04_recommend")

TEST_DAYS = 7
DIMS = [None, 256, 64]


def split(interactions: pd.DataFrame, n_embedded: int):
    cutoff = interactions.t_dat.max() - pd.Timedelta(days=TEST_DAYS)
    embedded = interactions[interactions.product_idx < n_embedded]
    return embedded[embedded.t_dat <= cutoff], embedded[embedded.t_dat > cutoff], cutoff


def popular_before(train: pd.DataFrame, cutoff) -> np.ndarray:
    recent = train[train.t_dat > cutoff - pd.Timedelta(days=TEST_DAYS)]
    return recent.product_idx.value_counts().index.to_numpy()


def popularity_recommend(ranking: np.ndarray, histories: list[list[int]], k: int) -> np.ndarray:
    return np.array([[p for p in ranking[: k + len(h)] if p not in set(h)][:k] for h in histories])


def score(recs: np.ndarray, answers: list[set[int]], k: int) -> dict[str, float]:
    hits = [len(set(r) & a) for r, a in zip(recs, answers)]
    return {
        "HitRate": np.mean([h > 0 for h in hits]),
        "Recall": np.mean([h / len(a) for h, a in zip(hits, answers)]),
        "Precision": np.mean([h / k for h in hits]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=12)
    parser.add_argument("--customers", type=int, default=2000)
    parser.add_argument("--history", type=int, default=20, help="최근 구매 몇 개로 평균을 낼지")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_embedded = len(load_embeddings())
    train, test, cutoff = split(pd.read_parquet(INTERACTIONS), n_embedded)
    histories = recommender.recent_history(train, n_embedded, args.history)
    bought_before = train.groupby("customer_idx").product_idx.agg(set)

    answers = {}
    for c, items in test.groupby("customer_idx").product_idx:
        if c in histories.index:
            new = set(items) - bought_before[c]
            if new:
                answers[c] = new
    rng = np.random.default_rng(args.seed)
    customers = rng.choice(sorted(answers), size=min(args.customers, len(answers)), replace=False)
    hist = [histories[c] for c in customers]
    ans = [answers[c] for c in customers]

    print(f"임베딩된 상품 {n_embedded:,}개 / 학습 기간 ~{cutoff.date()} / 정답 기간 이후 {TEST_DAYS}일")
    print(f"평가 고객 {len(customers):,}명 (조건을 만족한 고객 {len(answers):,}명 중), 고객당 정답 평균 {np.mean([len(a) for a in ans]):.1f}개\n")

    rows = {"인기순 (직전 7일)": score(popularity_recommend(popular_before(train, cutoff), hist, args.k), ans, args.k)}
    for dim in DIMS:
        matrix = load_embeddings(dim)
        rows[f"임베딩 {matrix.shape[1]}차원"] = score(recommender.recommend(matrix, hist, args.k), ans, args.k)

    print(f"{'방식':<18}{'HitRate@' + str(args.k):>12}{'Recall@' + str(args.k):>12}{'Precision@' + str(args.k):>14}")
    for name, s in rows.items():
        print(f"{name:<18}{s['HitRate']:>12.4f}{s['Recall']:>12.4f}{s['Precision']:>14.4f}")


if __name__ == "__main__":
    main()

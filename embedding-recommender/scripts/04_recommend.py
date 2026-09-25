"""4~5단계: 구매 목록의 평균 벡터로 Top-k 추천.

1. 사용자가 산 상품들의 벡터를 평균 낸다.
2. 평균 벡터를 다시 L2 정규화한다. 평균을 내면 길이가 1보다 짧아지기 때문(3-2강 p8).
3. 모든 상품과 코사인 유사도(= 정규화된 벡터끼리의 내적)를 구한다.
4. 이미 산 상품을 빼고 점수가 높은 k개를 추천한다.

사용 예
  python scripts/04_recommend.py                    # data/users.json의 모든 사용자
  python scripts/04_recommend.py --user home_cook   # 한 사용자
  python scripts/04_recommend.py --ids 52,64,53     # 구매 목록 직접 지정
  python scripts/04_recommend.py --dim 256          # MRL: 앞 256차원만 써서 비교
"""

import argparse
import json

import numpy as np

from common import EMBEDDINGS, EMBEDDINGS_META, PRODUCTS_FLAT, USERS, l2_normalize, read_jsonl


def load(dim: int | None):
    matrix = np.load(EMBEDDINGS)
    if dim:
        # MRL: 앞 차원만 잘라낸 뒤 반드시 재정규화(3-2강 p8)
        matrix = l2_normalize(matrix[:, :dim])
    ids = json.loads(EMBEDDINGS_META.read_text(encoding="utf-8"))["ids"]
    products = {r["id"]: r for r in read_jsonl(PRODUCTS_FLAT)}
    return matrix, ids, products


def recommend(matrix: np.ndarray, ids: list[int], purchased: list[int], k: int) -> list[tuple[int, float]]:
    index = {pid: i for i, pid in enumerate(ids)}
    unknown = [p for p in purchased if p not in index]
    if unknown:
        raise SystemExit(f"없는 상품 id: {unknown}")
    user_vector = l2_normalize(matrix[[index[p] for p in purchased]].mean(axis=0))
    scores = matrix @ user_vector
    scores[[index[p] for p in purchased]] = -np.inf
    top = np.argsort(-scores)[:k]
    return [(ids[i], float(scores[i])) for i in top]


def show(name: str, purchased: list[int], results, products) -> None:
    print(f"\n■ {name}")
    print("  구매:", " / ".join(f"{products[p]['title']} ({products[p]['category']})" for p in purchased))
    for rank, (pid, score) in enumerate(results, 1):
        p = products[pid]
        print(f"  {rank:>2}. {score:.4f}  [{pid:>3}] {p['title']}  ({p['category']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", help="data/users.json의 사용자 이름")
    parser.add_argument("--ids", help="쉼표로 구분한 구매 상품 id")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--dim", type=int, help="MRL 절단 차원(예: 256)")
    args = parser.parse_args()

    matrix, ids, products = load(args.dim)
    print(f"벡터: {matrix.shape[0]}개 × {matrix.shape[1]}차원, Top-{args.k}")

    if args.ids:
        users = {"custom": [int(x) for x in args.ids.split(",")]}
    else:
        users = json.loads(USERS.read_text(encoding="utf-8"))
        if args.user:
            users = {args.user: users[args.user]}
    for name, purchased in users.items():
        show(name, purchased, recommend(matrix, ids, purchased, args.k), products)


if __name__ == "__main__":
    main()

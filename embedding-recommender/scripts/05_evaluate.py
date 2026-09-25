"""평가: 하나 빼고 맞히기(leave-one-out)로 추천 품질을 숫자로 잰다.

1. 가상 사용자 생성: 카테고리마다 "그 카테고리 상품 3개를 산 사용자"를 여러 명 만든다.
2. 구매한 3개 중 1개를 숨기고, 나머지 2개로 Top-k를 추천한다(04_recommend.py와 같은 방식).
3. 점수
   - Hit@k: 숨긴 상품이 Top-k 안에 들어온 비율
   - 카테고리 일치율: 추천 k개 중 구매 상품과 같은 카테고리인 비율
4. 전체 차원과 MRL 절단 차원(256·128·64)을 한 표로 비교한다.

한계: "같은 카테고리를 산다"는 가정의 평가라 카테고리를 넘나드는 추천(라켓 → 운동화)은 재지 못한다.

사용 예
  python scripts/05_evaluate.py
  python scripts/05_evaluate.py -k 10 --users-per-category 50
"""

import argparse
import importlib
import itertools
import random
from collections import defaultdict

recommender = importlib.import_module("04_recommend")

DIMS = [None, 256, 128, 64]
BASKET_SIZE = 3


def make_users(products: dict, per_category: int, seed: int) -> list[tuple[str, tuple[int, ...]]]:
    by_category = defaultdict(list)
    for pid, p in products.items():
        by_category[p["category"]].append(pid)
    rng = random.Random(seed)
    users = []
    for category, pids in sorted(by_category.items()):
        baskets = list(itertools.combinations(sorted(pids), BASKET_SIZE))
        for basket in rng.sample(baskets, min(per_category, len(baskets))):
            users.append((category, basket))
    return users


def evaluate(dim: int | None, users, k: int):
    matrix, ids, products = recommender.load(dim)
    hits = defaultdict(list)
    category_match = []
    for category, basket in users:
        for hidden in basket:
            rest = [p for p in basket if p != hidden]
            top = [pid for pid, _ in recommender.recommend(matrix, ids, rest, k)]
            hits[category].append(hidden in top)
            category_match.append(sum(products[pid]["category"] == category for pid in top) / k)
    all_hits = [h for hs in hits.values() for h in hs]
    per_category = {c: sum(hs) / len(hs) for c, hs in hits.items()}
    return matrix.shape[1], sum(all_hits) / len(all_hits), sum(category_match) / len(category_match), per_category, len(all_hits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--users-per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, _, products = recommender.load(None)
    users = make_users(products, args.users_per_category, args.seed)

    results = [evaluate(dim, users, args.k) for dim in DIMS]
    print(f"가상 사용자 {len(users)}명 × 숨김 {BASKET_SIZE}회 = 시행 {results[0][4]}회, Top-{args.k}\n")
    print(f"{'설정':<10}{'Hit@' + str(args.k):>8}{'카테고리 일치율':>16}")
    for dim, hit, match, _, _ in results:
        print(f"{str(dim) + '차원':<10}{hit:>8.3f}{match:>16.3f}")

    _, _, _, per_category, _ = results[0]
    print(f"\n카테고리별 Hit@{args.k} ({results[0][0]}차원, 낮은 순)")
    for category, hit in sorted(per_category.items(), key=lambda x: x[1]):
        print(f"  {category:<22}{hit:.3f}")


if __name__ == "__main__":
    main()

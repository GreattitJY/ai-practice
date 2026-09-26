"""2단계: 각 레코드를 자연어로 평탄화.

표의 한 행을 임베딩 모델이 읽기 좋은 문장 하나로 바꾼다.

설계 원칙
- 넣는 것: 상품이 "무엇인지"를 말하는 필드(이름, 카테고리, 브랜드, 태그, 설명)
- 숫자는 말로 바꾼다: 임베딩은 12.99와 14.99의 차이를 잘 구분하지 못하므로
  가격은 같은 카테고리 안에서의 가격대(budget / mid-range / premium)로 표현한다.
- 빼는 것: 평점·할인율·재고처럼 "무엇인지"와 무관한 필드.
  넣으면 성격이 전혀 다른 상품이 평점이 같다는 이유로 가까워질 수 있다.
"""

import csv
import json
from collections import defaultdict

from common import PRODUCTS_CSV, PRODUCTS_FLAT

CATEGORY_NAMES = {
    "mens-shirts": "men's shirts",
    "mens-shoes": "men's shoes",
    "mens-watches": "men's watches",
    "womens-bags": "women's bags",
    "womens-dresses": "women's dresses",
    "womens-jewellery": "women's jewellery",
    "womens-shoes": "women's shoes",
    "womens-watches": "women's watches",
    "tops": "women's tops",
}


def category_name(slug: str) -> str:
    return CATEGORY_NAMES.get(slug, slug.replace("-", " "))


def price_bands(rows: list[dict]) -> dict[str, str]:
    """카테고리별로 가격 순위를 3등분해 가격대 이름을 붙인다."""
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    bands = {}
    for items in by_category.values():
        items.sort(key=lambda r: float(r["price"]))
        n = len(items)
        for i, r in enumerate(items):
            bands[r["id"]] = ("budget", "mid-range", "premium")[min(3 * i // n, 2)]
    return bands


def flatten(row: dict, band: str) -> str:
    category = category_name(row["category"])
    brand = f" by {row['brand']}" if row["brand"] else ""
    tags = ", ".join(t for t in row["tags"].split(";") if t)
    return (
        f"{row['title']} is a {band} {category} product{brand}. "
        f"Tags: {tags}. "
        f"{row['description']}"
    )


def main() -> None:
    with PRODUCTS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bands = price_bands(rows)
    with PRODUCTS_FLAT.open("w", encoding="utf-8") as f:
        for r in rows:
            record = {"id": int(r["id"]), "title": r["title"], "category": r["category"], "text": flatten(r, bands[r["id"]])}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(rows)}개 문장 → {PRODUCTS_FLAT.name}\n예시:")
    for r in rows[:2]:
        print(" -", flatten(r, bands[r["id"]]))


if __name__ == "__main__":
    main()

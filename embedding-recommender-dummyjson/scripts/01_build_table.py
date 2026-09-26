"""1단계: 제품 테이블 입수.

DummyJSON(https://dummyjson.com, MIT 라이선스)의 상품 194개를 받아
추천에 쓸 컬럼만 남긴 CSV 테이블로 만든다.
리뷰·바코드·재고 같은 운영용 필드는 버린다.
"""

import csv
import json
import urllib.request

from common import PRODUCTS_CSV, RAW_PRODUCTS

SOURCE_URL = "https://dummyjson.com/products?limit=0"
COLUMNS = ["id", "title", "brand", "category", "price", "discount_percentage", "rating", "tags", "description"]


def fetch_raw() -> list[dict]:
    if not RAW_PRODUCTS.exists():
        print(f"다운로드: {SOURCE_URL}")
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as res:
            RAW_PRODUCTS.write_bytes(res.read())
    return json.loads(RAW_PRODUCTS.read_text(encoding="utf-8"))["products"]


def main() -> None:
    products = fetch_raw()
    with PRODUCTS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for p in products:
            writer.writerow({
                "id": p["id"],
                "title": p["title"],
                "brand": p.get("brand") or "",
                "category": p["category"],
                "price": p["price"],
                "discount_percentage": p["discountPercentage"],
                "rating": p["rating"],
                "tags": ";".join(p.get("tags", [])),
                "description": p["description"],
            })
    print(f"{len(products)}개 상품 → {PRODUCTS_CSV.relative_to(PRODUCTS_CSV.parent.parent)}")


if __name__ == "__main__":
    main()

"""2단계: 각 레코드를 자연어로 평탄화.

상품 한 행을 임베딩 모델이 읽기 좋은 영어 문장 하나로 바꾼다.

설계 원칙
- 넣는 것: 옷이 "무엇인지"를 말하는 필드(이름, 종류, 대상·코너, 소재 분류, 무늬, 설명)
- 빼는 것
  - 색상: 상품 단위로 묶었기 때문에 색상은 "이 상품의 성격"이 아니라 "고를 수 있는 옵션"이다.
    넣으면 종류가 전혀 다른 옷이 같은 색 목록을 가졌다는 이유로 가까워진다.
  - 판매량·색상 버전 수: 옷의 성격과 무관하다. 판매량은 추천 순서가 아니라 평가의 인기순 기준선에 쓴다.
- 중복 줄이기: product_group(예: Garment Upper body)과 product_type(예: Vest top)은
  함께 쓰면 뜻이 겹치지만, 상위 분류를 알려 주는 역할이 있어 짧게 괄호로 붙인다.
"""

import json

import pandas as pd

from common import PRODUCTS, PRODUCTS_FLAT


def flatten(p) -> str:
    # 관사(a/an)는 붙이지 않는다. "a trousers", "a all over pattern ..."처럼 어색해지는 경우가 많다.
    pattern = f"{p.pattern.lower()} " if p.pattern and p.pattern not in ("Solid", "Other pattern", "Unknown") else ""
    parts = [
        f"{p.prod_name}: {pattern}{p.product_type.lower()} ({p.product_group.lower()})",
        f"from H&M {p.index_name}, {p.section} section, {p.garment_group} line.",
    ]
    text = " ".join(parts)
    if p.detail_desc:
        text += f" {p.detail_desc}"
    return text


def main() -> None:
    products = pd.read_parquet(PRODUCTS)
    with PRODUCTS_FLAT.open("w", encoding="utf-8") as f:
        for p in products.itertuples():
            f.write(json.dumps({"product_idx": p.product_idx, "text": flatten(p)}, ensure_ascii=False) + "\n")
    print(f"{len(products):,}개 문장 → {PRODUCTS_FLAT.name}\n예시(판매량 상위):")
    for p in products.head(3).itertuples():
        print(" -", flatten(p))


if __name__ == "__main__":
    main()

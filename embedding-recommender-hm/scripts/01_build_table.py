"""1단계: 제품 테이블 입수.

H&M Personalized Fashion Recommendations(Kaggle) 데이터로 두 테이블을 만든다.

1. 상품 테이블: articles.csv의 색상 버전(article_id, 105,542개)을 상품(product_code, 47,224개)으로 묶는다.
   같은 상품이면 색상이 달라도 설명이 같아서, 색상 버전마다 임베딩하면 거의 같은 벡터가 여러 개 생긴다.
   판매량이 많은 순으로 정렬해 상품 번호(product_idx)를 0부터 붙인다.
2. 구매 테이블: 구매 기록(3,100만 건)의 article_id를 상품 번호로, customer_id(64자 문자열)를
   정수 번호로 바꿔 저장한다. 문자열을 그대로 두면 메모리를 수 GB 쓴다.

사전 준비: data/raw/에 articles.csv, transactions.parquet
"""

import json

import pandas as pd

from common import CUSTOMERS, INTERACTIONS, PRODUCTS, RAW_ARTICLES, RAW_TRANSACTIONS


def mode(s: pd.Series) -> str:
    return s.mode().iloc[0]


def build_products(articles: pd.DataFrame, sales: pd.Series) -> pd.DataFrame:
    g = articles.groupby("product_code")
    products = pd.DataFrame({
        "prod_name": g.prod_name.agg(mode),
        "product_type": g.product_type_name.agg(mode),
        "product_group": g.product_group_name.agg(mode),
        "index_name": g.index_name.agg(mode),
        "section": g.section_name.agg(mode),
        "garment_group": g.garment_group_name.agg(mode),
        # 무늬는 색상 버전마다 다를 수 있다. 모두 같을 때만 상품의 성격으로 본다.
        "pattern": g.graphical_appearance_name.agg(lambda s: s.iloc[0] if s.nunique() == 1 else ""),
        "colours": g.colour_group_name.agg(lambda s: ";".join(sorted(s.unique()))),
        "variants": g.size(),
        "detail_desc": g.detail_desc.agg(lambda s: s.dropna().iloc[0] if s.notna().any() else ""),
    })
    products["sales"] = sales.reindex(products.index).fillna(0).astype(int)
    products = products.sort_values("sales", ascending=False, kind="stable").reset_index()
    products.insert(0, "product_idx", range(len(products)))
    return products


def main() -> None:
    articles = pd.read_csv(RAW_ARTICLES, dtype={"article_id": str})
    print(f"색상 버전 {len(articles):,}개 읽음")

    transactions = pd.read_parquet(RAW_TRANSACTIONS, columns=["t_dat", "customer_id", "article_id"])
    print(f"구매 기록 {len(transactions):,}건 읽음")
    to_product = articles.set_index("article_id").product_code
    transactions["product_code"] = transactions.article_id.map(to_product)

    products = build_products(articles, transactions.product_code.value_counts())
    products.to_parquet(PRODUCTS, index=False)
    print(f"상품 {len(products):,}개 → {PRODUCTS.name}")

    code_to_idx = pd.Series(products.product_idx.values, index=products.product_code)
    customer_codes, customer_ids = pd.factorize(transactions.customer_id)
    interactions = pd.DataFrame({
        "t_dat": transactions.t_dat,
        "customer_idx": customer_codes.astype("int32"),
        "product_idx": transactions.product_code.map(code_to_idx).astype("int32"),
    })
    interactions.to_parquet(INTERACTIONS, index=False)
    CUSTOMERS.write_text(json.dumps(customer_ids.tolist()))
    print(f"구매 {len(interactions):,}건, 고객 {len(customer_ids):,}명 → {INTERACTIONS.name}")


if __name__ == "__main__":
    main()

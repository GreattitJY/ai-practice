# 임베딩 기반 상품 추천 (H&M)

"현대 AI의 원리와 구조" 3-2강(임베딩과 벡터의 원리) 실습.
H&M의 실제 상품·구매 데이터로, 고객이 산 상품들의 평균 벡터와 가까운 상품을 추천하고
다음 주에 실제로 산 상품을 맞히는지 평가한다.

작은 데이터로 한 같은 실습: `../embedding-recommender-dummyjson`

## 데이터 받기

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
대회 데이터를 쓴다. 대회 규정(Rules)에 동의해야 받을 수 있고, 사용 조건 때문에 데이터는 저장소에 올리지 않는다(`.gitignore`).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/kaggle auth login     # 브라우저 로그인

cd data/raw
C=h-and-m-personalized-fashion-recommendations
../../.venv/bin/kaggle competitions download -c $C -f articles.csv -p .
../../.venv/bin/kaggle competitions download -c $C -f transactions_train.csv -p .
unzip articles.csv.zip && unzip transactions_train.csv.zip && rm *.zip
cd ../..
.venv/bin/python -c "import pandas as pd; pd.read_csv('data/raw/transactions_train.csv', dtype={'article_id': str}, usecols=['t_dat','customer_id','article_id'], parse_dates=['t_dat']).to_parquet('data/raw/transactions.parquet')"
```

이미지 파일(수십 GB)이 포함돼 있으니 대회 데이터 전체를 한 번에 받지 말 것.

## 실행

| 단계 | 명령 | 결과 |
|---|---|---|
| 1. 제품 테이블 | `.venv/bin/python scripts/01_build_table.py` | 상품 47,224개(색상 버전을 상품 단위로 묶음), 구매 3,179만 건 |
| 2. 자연어 평탄화 | `.venv/bin/python scripts/02_flatten.py` | `data/products_flat.jsonl` |
| 3. 벡터 변환 | `caffeinate -i .venv/bin/python scripts/03_embed.py --limit 10000` | 판매량 상위부터. 멈춰도 다시 실행하면 이어서 함 |
| 4~5. 추천 | `.venv/bin/python scripts/04_recommend.py` | 무작위 고객 3명의 추천. `--customer`, `--product`, `--dim` |
| 평가 | `.venv/bin/python scripts/05_evaluate.py` | 마지막 7일 실제 구매 기준 HitRate·Recall·Precision, 인기순과 비교 |

3단계는 LM Studio 로컬 서버가 필요하다: `lms server start`, `lms load text-embedding-qwen3-embedding-4b`.
M1 Pro 기준 초당 약 3.8건(1만 개 약 45분, 전체 약 3.5시간).

## 설계 메모

- **상품 단위**: 색상 버전(105,542개)마다 설명이 같아서, 상품(product_code) 단위로 묶어 임베딩한다.
- **평탄화**: 이름·종류·대상 코너·소재 분류·무늬·설명을 문장으로 만든다. 색상은 상품의 성격이 아니라 고를 수 있는 옵션이라 뺀다.
- **지시문 없음**: 상품과 상품을 비교하는 대칭 구조라 `Instruct:` 접두어를 붙이지 않는다(3-2강 p5).
- **정규화**: 상품 벡터와 평균 벡터를 모두 L2 정규화한다(3-2강 p8, p15).
- **평가**: 이미 산 상품은 추천에서 빼므로, 정답도 "처음 산 상품"만 센다.

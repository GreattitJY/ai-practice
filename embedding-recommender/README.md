# 임베딩 기반 상품 추천

"현대 AI의 원리와 구조" 3-2강(임베딩과 벡터의 원리) 실습.
구매한 상품들의 평균 벡터와 가까운 상품을 Top-k로 추천한다.

## 준비

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
lms server start
lms load text-embedding-qwen3-embedding-4b
```

## 실행

| 단계 | 명령 | 결과 |
|---|---|---|
| 1. 제품 테이블 입수 | `.venv/bin/python scripts/01_build_table.py` | `data/products.csv` (DummyJSON 194개, MIT) |
| 2. 자연어 평탄화 | `.venv/bin/python scripts/02_flatten.py` | `data/products_flat.jsonl` |
| 3. 벡터 변환 | `.venv/bin/python scripts/03_embed.py` | `data/embeddings.npy` (194 × 2560) |
| 4~5. 평균 벡터 → Top-k | `.venv/bin/python scripts/04_recommend.py` | 터미널 출력 |

`04_recommend.py` 옵션: `--user home_cook`, `--ids 52,64,53`, `-k 10`, `--dim 256`(MRL 절단)

샘플 사용자의 구매 목록은 `data/users.json`에 있다.

## 설계 메모

- **평탄화**: 이름·카테고리·브랜드·태그·설명만 문장으로 만든다. 가격은 숫자 대신 카테고리 안에서의 가격대(budget / mid-range / premium)로 쓴다. 평점·할인율·재고는 상품의 성격과 무관해서 뺀다.
- **지시문 없음**: 상품과 상품을 비교하는 대칭 구조라 검색용 `Instruct:` 접두어를 붙이지 않는다(3-2강 p5).
- **정규화**: 상품 벡터와 평균 벡터를 모두 L2 정규화한다. 그래서 내적이 곧 코사인 유사도다(3-2강 p8, p15).
- **이미 산 상품은 제외**하고 추천한다.

## 데이터 출처

`data/`의 상품 데이터는 [DummyJSON](https://github.com/Ovi/DummyJSON)(MIT)에서 받은 것이다. 라이선스 원문은 `data/LICENSE-DummyJSON`에 있다.

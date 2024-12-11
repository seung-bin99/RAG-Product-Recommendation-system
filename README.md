# RAG-Product-Recommendation-system
> - 현재 진행중
## 사용자가 질문하면 필요한 제품을 추천해주는 RAG 추천 시스템
### 캐글의 Amazon sales & E-commerce behavior dataset 사용

### Process
- 캐글에서 적절한 사용자 행동 패턴 및 쇼핑몰 로그 데이터를 수집함
- E-commerce 쇼핑몰 로그 데이터를 이커머스 분석 기법 리서치 및 적용
- Amazon 사용자 행동 데이터 전처리
- 각 전처리 된 데이터를 임베딩 모델을 사용해 벡터 임베딩 한 후, Chroma나 FAISS 벡터 데이터베이스에 저장
- LLM 모델을 사용해 벡터 데이터베이스 기반 QnA 챗봇 개발 및 검증

#### 문제 정의
- 쇼핑 플랫폼 운영 중, 매출이 줄고있음
#### 가설 설정
- 가설 1 : 고객들이 해당 플랫폼에 방문율이 감소하고 있을 것이다.(X)
- 가설 2 : 고객들이 제품을 살펴보지만, 높은 가격으로 인해 구매를 하지않을 것이다.(O)

#### 데이터 수집
- 캐글의 Amazon sales & E-commerce behavior dataset 다운

#### 역할 세부 내용
1) 캐글에서 Amazon sales & E-commerce behavior dataset 수집
2) E-commerce 쇼핑몰 로그 데이터 기반 DAU 계산, 퍼널분석, Kruskal-Wallis 분석거쳐 솔루션 아이디어 도출
- 솔루션 적용 단계
3) E-commerce 쇼핑몰 로그 데이터 RFM 분석
> - 13개의 고객 그룹 segmentation
4) E-commerce 데이터를 유저 상태에 따라 전처리
> - 구매 유저 : 세션 기간에 따른 clustering + RFM 분석 적용 고객 세분화
> - 구매하지 않은 유저 : 세션 기간에 따른 clustering
5) E-commerce 데이터와 유사한 데이터 처리위해 Amazon 사용자 행동 데이터에서 필요한 변수 전처리
> - 카테고리 / 제품 id / 제품명 / 할인가 / 원가 추출
6) BAAI/bge-m3 임베딩 모델을 사용해 두개의 데이터를 벡터 임베딩 한 후, Chroma 벡터 데이터베이스에 저장
7) Chroma에서 임베딩 벡터 불러와 MMR 검색 방식의 Retriever 생성
8) Gemma-7b LLM 모델을 활용해 Prompt 설정 후, RAG Chain 생성해 사용자 질문에 문장으로 제품을 추천하도록 구현
9) MRR과 평균 코사인 유사도로 성능 검증
> - Predict 데이터 셋 : Amazon 임베딩 벡터
> - 검증 데이터 셋 : E-commerce 임베딩 벡터

### LLM 아키텍처
![image](https://github.com/user-attachments/assets/f14feb94-4493-408b-ace5-8011a208f673)

### 분석 과정
![퍼널분석](https://github.com/user-attachments/assets/d647a434-bc4e-4ebf-83f7-3fc65878d05b)
![세 집단 비교 분석](https://github.com/user-attachments/assets/661ec491-99bb-4031-b3c1-87e0463fb5f8)
![RFM 분포 확인](https://github.com/user-attachments/assets/4f86c992-bac9-4b1d-b898-beb5991db79f)
![RFM 분석 적용한 고객 Segmentation](https://github.com/user-attachments/assets/38f7c369-d06c-49bb-a8dc-77eefee0086b)


### RAG 생성 답변 성능 검증
![image](https://github.com/user-attachments/assets/15562f48-f677-49c2-bea4-3e89d8e4834f)
- 성능 검증 진행과정
1. 결과 수집: rag_chain_amazon과 rag_chain_reference를 통해 예측 및 검증 결과를 수집.
2. 문장 분리: 예측 결과와 검증 결과를 문장 단위로 나눔.
3. 임베딩 생성: 각 문장을 임베딩하여 벡터로 변환.
4. MRR 및 코사인 유사도 계산: 각 예측 문장에 대해 검증 문장과의 유사도를 계산하고, 관련성이 높은 문장의 순위를 반영하여 MRR을 업데이트. 최대 유사도를 누적하여 평균 코사인 유사도를 계산.
> - Predict Result, Reference Result 답변 문장 결과 기반 성능 평가 시, 어느 정도 고른 성능 결과를 볼 수 있음.

#### 성능 지표

- Mean Reciprocal Rank (MRR) : 사용자의 쿼리에 대한 가장 관련성 높은 결과의 순위의 역수를 평균한 값.

![image (4)](https://github.com/user-attachments/assets/3846d604-b3af-46ab-8ce9-14e6298c2e17)
> - 계산 방법: 각 쿼리에서 가장 관련성이 높은 결과의 순위를 찾고, 그 순위의 역수를 구함. 여러 쿼리에 대해 평균을 계산.
> - 의미: MRR이 1에 가까울수록 시스템이 관련성 높은 결과를 잘 제공함을 의미.


- 평균 코사인 유사도 (Average Cosine Similarity) : 예측된 문장과 참조 문장 간의 유사성을 측정하는 지표.
  
![image (3)](https://github.com/user-attachments/assets/4f032a1b-4b0a-494b-9a00-839b74d123a0)
> - 계산 방법: 각 예측 문장에 대해 모든 참조 문장과의 코사인 유사도를 계산. 각 예측 문장에 대해 최대 유사도를 찾아 총합. 쿼리 수로 나누어 평균을 구함.
> - 의미: 평균 코사인 유사도가 1에 가까울수록 예측과 참조 문장이 유사함을 나타냄.

### 사용 Dataset
#### Amazon sales 데이터
> **[Amazon sales dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)**

#### E-commerce behavior 데이터
> **[E-commerce behavior dataset](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)**

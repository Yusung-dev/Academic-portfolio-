## 📄 논문 정리: NC-net: Neighbourhood Consensus Networks

> 논문 링크: https://arxiv.org/abs/1810.10510
> 
> 발표 학회/연도: NeurIPS 2018 (NIPS 2018)
> 
> 논문 저자: Ignacio Rocco, Mircea Cimpoi, Relja Arandjelović, Akihiko Torii, Tomas Pajdla, Josef Sivic
<br>

**해당 논문 선택 이유**


---

### 이 논문의 핵심 아이디어는 무엇인가?
### 이 연구가 중요한 이유는 무엇인가?
### 기존 연구들의 한계는 무엇인가?
### 그 한계를 어떻게 해결하였는가?
### 제안 방법의 구조는 어떤가?
### 어떤 성과를 얻었는가?
### 어떤 데이터를 사용했는가?
### 비판적 읽기 & 한계점은 무엇인가?
### 한줄 요약 및 개인적 생각

### 직관적으로 구조 이해하기
1. Dense CNN feature extraction
    - 입력 이미지를 ResNet기반 CNN을 통과시켜서 feature map을 얻는다: 픽셀단위 값이 아닌 각 위치마다 의미 있는 고차원 표현 벡터로 바꿈 = 이후 노이즈에 덜 취약

2. 4D Feature Matching space (correlation map)
    - 두 이미지의 feature map 를 곱하여(정규화 상관) 유사도 기반 correlation을 구한다 : 모든 픽셀 쌍의 상관관계(유사도)를 나타내는 map으로 source 이미지 한 pixel과 target 이미지의 어떤점과 닮았는지 전부 보여주는 heatmap, 모든 가능한 모든 매칭 후보
 
3. soft mutual nearest neighbour filtering (SMNN)
    - corr map에서 뽑은 상관관계 유사도에서 사진 A의 (i,j) -> 사진 B의 (k,l) 신뢰도 있을거고 사진 B의 (k,l) -> 사진 A의 (i,j)의 신뢰도가 있을텐데 양쪽에서 서로 선택한 매칭만 신뢰하겠다는 규칙임
    - 만약 (i,j) <-> (k,l)이 서로의 best match 신뢰도라면 값이 1에 근사 서로 best match 신뢰도가 아니면 0에 근사
    - 최종적으로 일관되고 상호적인 매칭을 강화하고, 일방적이거나 불일치한 매칭이 약화됨

4. NC network
    - SMNN을 거친 corr map을 입력으로 넣음(SMNN으로 전처리된 모든 픽셀 쌍의 상관관계) 4d convolution을 사용함으로써
5. 4D filtered matches
    -
번외. Loss SMNN
    - 

### feed forward -> backpropagation 직관적 이해하기

### 궁금했던 부분

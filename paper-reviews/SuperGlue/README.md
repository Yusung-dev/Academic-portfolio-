## 📄 논문 정리: SuperGlue: Learning Feature Matching with Graph Neural Networks

> 논문 링크: https://arxiv.org/abs/1911.11763
> 
> 발표 학회/연도: CVPR 2020 (Oral)
> 
> 논문 저자: Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich (Magic Leap & ETH Zürich)
<br>

**해당 논문 선택 이유**


---

### 이 논문의 핵심 아이디어는 무엇인가?
- 두 이미지의 키포인트 집합을 그래프 노드로 보고,
Self-Attention(동일 이미지 내 문맥),
Cross-Attention(상대 이미지와의 상호참조)
로 매칭에 유효한 ‘문맥적 표현’(matching descriptors)을 만든 후,
**Sinkhorn 최적 수송(Optimal Transport)**으로 **부분 할당(Partial Assignment)**을 미분 가능하게 산출한다.
### 이 연구가 중요한 이유는 무엇인가?
### 기존 연구들의 한계는 무엇인가?
- SIFT/ORB + ratio test/상호검사 등은 강한 변환·저텍스처·반복 패턴에서 취약하다
- 딥 디스크립터(SuperPoint 등)도 최근접(ANN) 기반 매칭에 의존 → 글로벌 일관성 부재, outlier 정제에 추가 휴리스틱 필요하다
- 기존 학습식 inlier 분류(OANet 등)는 이미 제안된 매치 집합만 다뤄 할당 제약(1:1 매칭, 미매칭)을 내재화하지 못한
### 그 한계를 어떻게 해결하였는가?
- GNN + (Self/Cross) Multi-Head Attention으로 집합-대-집합 수준의 표현을 학습하여 전역 문맥을 반영한다
- Dustbin(미매칭용 가상 노드)과 Sinkhorn 반복 정규화로 부분 할당을 연속·미분 가능하게 근사하여 상호검사/ratio-test를 대체하는 구조적 제약을 학습 과정에 내재화한다
### 제안 방법의 구조는 어떤가?
 
   3. ### 어떤 성과를 얻었는가?
### 어떤 데이터를 사용했는가?
### 비판적 읽기 & 한계점은 무엇인가?
### 한줄 요약 및 개인적 생각

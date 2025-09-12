# 📚 Paper Reviews Workflow

안녕하세요! 이 폴더는 제가 논문을 체계적으로 읽고, 정리하며, 구현하고, 응용하는 과정을 기록하는 공간입니다.  
아래는 제가 논문 리뷰를 진행하는 전반적인 파이프라인입니다.

---

## 1. 논문 찾기 (What to read?)

- 대표 학회 논문 위주로 보았습니다
  (예: CVPR, ECCV, ICCV, AAAI, NeurIPS, WACV  등)  
- Arxiv-sanity, Arxiv Digest, HuggingFace Papers 같은 큐레이션 사이트를 적극 활용하였습니다

---

## 2. 논문 읽기 (How to read?)

- 논문 읽는 순서  
  1. 1차 읽기: 전체 구조 파악 (abstract → introduction → conclusion)  
  2. 2차 읽기: 깊게 읽기 (method [수식, 구조] → 실험 → related work)

- 논문 읽을 때 체크할 포인트  
  1. 이 논문의 핵심 아이디어는 무엇인가/어떤 문제를 해결하려 하는가?  
  2. 이 연구가 왜 중요한가?  
  3. 기존 연구들의 한계/빈틈은 무엇인가? (관련 연구들이 어떻게 발전해 왔는가)  
  4. 앞서 말한 한계를 어떻게 해결하였는가? (기존 방법과의 차별점/새로워진 점)  
  5. 구조/알고리즘은 어떻게 생겼는가?
  6. 새로운 방법으로 어떤 성과가 나왔는가?
  7. 연구에 사용된 데이터셋 혹은 자료는 무엇인가? (공개/수집 데이터, 데이터 규모, 특징)  
  8. 그럼에도 논문에서 해결되지 않은 한계점은 무엇인가?

---

## 3. 논문 정리 (How to digest?)

논문 제목 :

1. 이 논문의 핵심 아이디어는 무엇인가?
2. 이 연구가 중요한 이유는 무엇인가?
3. 기존 연구들의 한계는 무엇인가?
4. 그 한계를 어떻게 해결하였는가?
5. 제안 방법의 구조는 어떤가?
6. 어떤 성과를 얻었는가?
7. 어떤 데이터를 사용했는가인적 생각

---

## 4. 논문 구현 및 재현 (Reproduce)

- Papers with Code에서 공식 코드나 재구현 코드를 clone 후 논문 실험을 재현하였습니다
- (작은 데이터셋으로 재현하였습니다)  
- 경량 버전/디버그용 코드가 있다면 그것을 이용하였습니다  
- 논문 전체를 돌리기 어렵다면 핵심 알고리즘 모듈만 따로 구현하고 실험해 보았습니다

---

## 5. 논문 비교 및 실험 (Experiment & Evaluate)

- 논문 구조 일부 바꿔보기 (예: Activation, Optimizer)  
- 다른 데이터셋에 적용해보기  
- Ablation Study 따라하거나 반대로 해보기  

---

## 6. 주제 응용 및 발전 (Extend or Apply)

- 기존 논문을 다른 task에 적용해 보기 
- 여러 논문 아이디어를 조합해 보기  
- 본인의 관심 분야 문제에 맞춰 실험해 보기  
- Kaggle, Dacon 대회 등에 적용해 보기  

---

## 7. 자기 주제 정립 (Research Topic Discovery)

- 부족한 부분이 보이면 필요한 추가 연구를 수행하였습니다
- 그 부분이 논문으로 발전할 가능성이 보인다면 본격적인 연구주제로 확장하였습니다.

---

감사합니다!  

## 논문 목록

1. [Image Style Transfer Using Convolutional Neural Networks](./ImageStyleTransfer_CNN/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 정리](./ImageStyleTransfer_CNN/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 구현](./ImageStyleTransfer_CNN/구현)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 응용](./ImageStyleTransfer_CNN/응용) 

2. [**KeyNet**: Keypoint Detection by Handcrafted and Learned CNN filters](./KeyNet/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 정리](./KeyNet/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 구현](./KeyNet/구현)  
   
3. [**SuperPoint**: Self-Supervised Interest Point Detection and Description](./SuperPoint/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 정리](./SuperPoint/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 구현](./SuperPoint/구현)  

4. [**SuperGlue**: Learning Feature Matching with Graph Neural Networks](./SuperGlue/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 정리](./SuperGlue/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 구현](./SuperGlue/구현)  

5. [**LoFTR**: Detector-Free Local Feature Matching with Transformers](./LoFTR/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 정리](./LoFTR/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 구현](./LoFTR/구현)  

6. [**GeoCNN**: Convolutional Neural Network Architecture for Geometric Matching](./GeoCNN/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 정리](./GeoCNN/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 구현](./GeoCNN/구현)  

7. [**Weakalign**: End-to-end weakly-supervised semantic alignment](./Weakalign/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 정리](./Weakalign/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 구현](./Weakalign/구현)  

8. [**NC-net**: Neighbourhood Consensus Networks](./NCnet/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 정리](./NCnet/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 구현](./NCnet/구현)

9. [**Hyperpixel Flow**: Semantic Correspondence with Multi-layer Neural Features](./HyperpixelFlow/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 정리](./HyperpixelFlow/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 구현](./HyperpixelFlow/구현)

10. [**SD4Match**: Stable Diffusion Features for Semantic Matching](./SD4Match/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 정리](./SD4Match/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 구현](./SD4Match/구현)

11. [**SD4Match**: Stable Diffusion Features for Semantic Matching](./SD4Match/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 정리](./SD4Match/README.md)  
    &nbsp;&nbsp;&nbsp;&nbsp;- [논문 구현](./SD4Match/구현)

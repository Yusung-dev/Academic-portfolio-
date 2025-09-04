## 📄 논문 정리: End-to-End Weakly-Supervised Semantic Alignment

> 논문 링크: https://arxiv.org/abs/1712.06861
> 
> 발표 학회/연도: CVPR 2018
> 
> 논문 저자: Ignacio Rocco, Relja Arandjelović, Josef Sivic
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


### 구조 직관적으로 이해하기


1. **특징 추출**
    - 입력 이미지를 ResNet기반 CNN을 통과시켜서 feature map을 얻는다: 픽셀단위 값이 아닌 각 위치마다 의미 있는 고차원 표현 벡터로 바꿈 = 이후 노이즈에 덜 취약
      
2. **쌍별 특징 매칭**
    - 두 이미지의 feature map 를 곱하여(정규화 상관) 유사도 기반 correlation을 구한다 : 모든 픽셀 쌍의 상관관계(유사도)를 나타내는 map으로 source 이미지 한 pixel과 target 이미지의 어떤점과 닮았는지 전부 보여주는 heatmap, 모든 가능한 모든 매칭 후보
      
3. **매칭 점수 공간**
    - 쌍별 특징 매칭의 결과물 S이며 가능한 모든 매칭 후보, 후보들 중에서 그럴듯한 매칭은 더 높은 점수를 가지는 map임
      
4. **기하 변환 추정**
    - corr map을 input으로 넣어 (B,2,3)짜리 theta를 뽑아냄 이 theta는 후에 loss에 의해 corr map을 보고 source 이미지에 기하학적 변환을 주어서 target 이미지에 잘 들어맞도록 (=soft inlier가 많도록) theta가 update됨
      
5. **인라이어 마스크 생성**
    - 추정된 theta를 가지고 corr map의 값은 그대로 두고 그 값을 참조하는 좌표 위치바꾸는 인라이어 맵을 생성
      
6. **마스킹된 매칭 점수**
    - 매칭 점수공간에 인라이어 확률 맵으로 가중치를 적용함으로써 outlier는 매칭이 무시되고, inlier는 강조됨
      
7. **소프트 인라이어 개수**
    - 최종적으로 soft inlier를 count하는 함수를 정의해 inlier의 수를 구함 이때 loss는 이 count가 커지도록 학습시킴

### 궁금했던 부분
- 결국에 pixel by pixel로 고양이 귀 - 고양이 귀를 찾는 문제인데(=resnet 통해서 feature map뽑고 그 vector가 비슷하면 아 서로 같은 고양이 귀- 고양이 귀 이겠다며 비교해서 찾으면 되는데, 즉, corr map까지만 필요해 보이는데 ) 사진을 affine하는 이유는 affine함으로써 backpropagtion때 사진 A의 고양이 귀 pixel의 vector와 사진 B의 고양이 귀 pixel의 vector가 이전 그냥 feature map뽑았을때 보다 더욱 비슷해진다
- affine하는게 귀옆에 눈이 있다라는 전역적 정보를 주기 위함이 아님. 정말 이 논문은 pixel by pixel로만 비교함

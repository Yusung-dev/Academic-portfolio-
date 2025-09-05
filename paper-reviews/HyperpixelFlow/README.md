## 📄 논문 정리: Hyperpixel Flow: Semantic Correspondence with Multi-layer Neural Features

> 논문 링크: https://arxiv.org/abs/1908.06537
> 
> 발표 학회/연도: ICCV 2019
> 
> 논문 저자: Junho Kim, Dahun Kim, Dongyoon Han, Sanghyuk Chun, Junsuk Choe, Bohyung Han
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
1. 하이퍼 픽셀 구성
    - 입력 이미지로 feature map을 얻는다, but CNN feature는 레이어 깊이에 따라 성질이 다르다
      - 얕은 레이어: dege, texture, local detail
      - 중간 레이어: part-level 구조 (귀, 눈, 바퀴 등)
      - 깊은 레이어: semantic context (이건 고양이다, 자동차다)
    - 그럼으로 한레이어 사용하면 semantic correspondences에서는 디테일도 필요하고 의미도 필요하기에 부족해서 레이어를 합치는 선택을 하였다
    - Beam search라는 방식으로 몇개의 layer를 뽑아내 feature map 해상도(H,W)를 맞추기 위해 upsampling해서 concate함으로써 여러 정보를 담은 hyperpixel feature map을 만들었다
    - Hyperpixel construction은 CNN 여러 레이어의 정보를 한 픽셀 단위로 합쳐서, 디테일+구조+의미를 모두 담은 강력한 표현을 만드는 과정이다

2. 정규화 하프 매칭
    - 먼저 사진 2개를 hyperpixeld feature의 cosine similarity를 함으로써 B,N,N으로 서로가 얼마나 비슷한지 상관관계 맵을 만든다
    - 단순히 서로의 feature vector가 닮았다고 매칭을 하게되면 오류가 많기에 기하학적 일관성을 반영하는 방식(Hough Voting)을 사용한다 좋은 매칭이라면 주변 매칭들(귀, 눈, 코)가 비슷한 방향으로 이동해야한다는 아이디어를 사용한다
    - Hough Voting을 사용해 offset을 ouput으로 내놓는다 offset은 source 고양이귀 -> target 고양이 귀 까지의 x,y변화량이다 즉, (B,H,W,2)이며 [B,H,W,0]이 source -> target으로 x변화량 [B,H,W,1]이 source -> target으로 y변화량이다

4. 플로우 형성
    - 정규화 하프 매칭을 통해 받은 B,W,H,2를 mathcing하는 역할을 한다
      
### feed forward -> backpropagation 직관적 이해하기
### 궁금했던 부분

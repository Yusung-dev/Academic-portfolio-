## 📄 논문 정리: Key.Net: Keypoint Detection by Handcrafted and Learned CNN filters

![result1](./assets/result.jpg)

> 논문 링크: https://arxiv.org/abs/1508.06576
> 
> 발표 학회/연도: ICCV 2019 (International Conference on Computer Vision 2019)
> 
> 논문 저자: Axel Barroso-Laguna, Edgar Riba, Daniel Ponsa, Krystian Mikolajczyk
<br>

**해당 논문 선택 이유**  
3d vision을 공부하기로 마음먹고 맨 처음에 무엇을 배워야하냐라고 물어본다면 
저는 wide-baseline matching을 공부해야한다고 생각합니다 그래서 wide-baseline matching중에서
detection분야에서 좋은 성능을 냈었던 KeyNet을 선택하게되었습니다

---

### 이 논문의 핵심 아이디어는 무엇인가?

- **핵심 아이디어** : 수작업(handcrafted) 도함수 필터와 학습된 CNN 필터를 얕은 멀티 스케일 아키텍처 안에서 결합해, 스케일 변화에도 반복성 높은 키포인트를 검출한다 이를 위해 멀티 스케일 인덱스 프로포절 손실로 스케일 전반에 걸쳐 안정적인 특징에 높은 점수를 주도록 학습한다 결과적으로 repeatability/매칭/복잡도에서 최고성능을 끌어냈다

### 이 연구가 중요한 이유는 무엇인가?

- 현실 응용(AR 헤드셋/스마트폰)에서 신뢰성과 효율성을 갖춘 희소(sparse) 로컬 특징 검출기 수요가 커졌지만, detector 쪽에선 학습 방법이 handcrafted 대비 확실한 우위를 보여주지 못했음 → 이 간극을 메우는 시도를 하였기에 중요하다

### 기존 연구들의 한계는 무엇인가?

- 완전 CNN 기반 detector들은 널리 쓰이는 repeatability 지표에서 개선이 제한적이었고, 특징 영역의 아핀 파라미터(특히 스케일) 추정 정확도가 낮다는 문제가 지적되어왔다 한편 커뮤니티는 한동안 희소 검출보다 밀집 표현/매칭에 치우치는 경향이 있었다

### 그 한계를 어떻게 해결하였는가?

- Handcrafted 필터를 ‘약한 앵커(soft anchor)’로 도입해 학습 안정성과 수렴을 높이고 파라미터 수를 대폭 축소하였으며 동시에 CNN 블록이 지역화/점수화/순위를 학습함으로써 해결하였다
- 멀티스케일 피라미드로 입력 3스케일(블러+1.2배 다운샘플)을 병렬 처리하고, 업샘플+컨캣 뒤 최종 컨볼루션으로 response map 산출하였다
- IP(Index Proposal) 레이어로 창(window)별 미분 가능한 soft‑argmax 위치 추출, 그리고 M‑SIP 손실로 다양한 창 크기/스케일에서 공변 제약을 평균해 스케일 전반에 지배적인 키포인트를 학습하도록 유도하였으며 학습은 이미지 간 기하 변환(호모그래피) 만으로 지도하였다

### 제안 방법의 구조는 어떤가?

<div style="display: flex; align-items: flex-start;">
  <img src="./assets/paper1.jpg" width="300" style="margin-right: 20px;">
</div>

>출처: Axel Barroso-Laguna, Key.Net, ICCV 2019

- 입력: 동일

### 어떤 성과를 얻었는가?
### 어떤 데이터를 사용했는가?
### 한계점은 무엇인가?
### 한줄 요약 및 갱니적 생각

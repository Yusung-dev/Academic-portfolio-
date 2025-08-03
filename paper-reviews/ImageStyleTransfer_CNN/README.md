## 📄 논문 정리: Image Style Transfer Using Convolutional Neural Networks

![result](/paper-reviews/ImageStyleTransfer_CNN/구현/assets/result.jpg)

> 논문 링크: https://arxiv.org/abs/1508.06576
> 
> 발표 학회/연도: CVPR 2016 (IEEE Conference on Computer Vision and Pattern Recognition)
> 
> 논문 저자: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

---

### 이 논문의 핵심 아이디어는 무엇인가?

객체 인식을 위해 최적화된 합성곱 신경망(Convolutional Neural Networks)에서 
유도된 이미지 표현을 사용하여 자연 이미지의 내용(content)와 스타일(style)을 
분리하고 결합할 수 있다 이를 통해 임의의 사진의 내용을 다양한 유명 예술작품의 
외형과 결합한 높은 지각적 품질을 가진 새로운 이미지를 생성할 수 있다


### 이 연구가 중요한 이유는 무엇인가?

기존의 연구에서는 전통적인 필터나 텍스처 합성 기법에 의존해왔다 
그에 반해 이 논문은 CNN(Convolutional Neural Network)을 이용하여 스타일과 콘텐츠를 분리하여
조합하하는 새로운 방법을 제한하였다


### 기존 연구의 한계는 무엇인가

기존 스타일 전송 방법들은 주로 수동적으로 필터나 텍스처 패턴을 조합하는 방식으로 스타일과
콘텐츠를 명확히 분리하지 못하였다


### 그 한계를 어떻게 해결하였는가?

기존 방식은 스타일과 콘텐츠를 분리해서 다루지 못했지만, 이 논문은 CNN의 계층적 특성 표현을 
활용하여 스타일과 콘텐츠를 명확히 분리하였다

### 제안 방법의 구조는 어떤가?

사전학습된 VGG19모델을 사용하였다 
![construction]("https://github.com/user-attachments/assets/7d9b064c-27ef-4bd7-9664-43737344c52d" )
출처: Gatys et al., Image Style Transfer Using CNNs, CVPR 2016


스타일이미지와 콘텐츠 이미지의 feature를 각각 추출하였다  
    content : vgg19의 특정 상위 계층(conv4_2)  
    style : vgg19의 여러 계층의 Gram matrix(conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)  
  
random noise이미지에서 시작해 스타일과 콘텐츠의 loss를 최소화하도록 이미지를 반복적으로
업데이트한다  
최종적으로 콘텐츠 구조는 유지하면서 스타일 특성이 반영된 결과 이미지를 생성한다


### 어떤 성과를 얻었는가?

예술 작품의 스타일을 사진에 자연스럽게 입힐 수 있게 되었다 그리고 이후 많은 후속 연구
(실시간 스타일 전송, GAN)의 기초적인 아이디어를 제공했을뿐만 아니라 예술, 디자인, 미디어 등
여러분야로 영향을 확대하였다


### 어떤 데이터를 사용했는가?

정형화된 학습데이터를 사용하진 않았다 대신 임의의 style사진, 임의의 content사진을 입력으로
사용하였다


### 한계점은 무엇인가?

직접 코드화해본 결과 계산량이 상당이 많다  
또한 논문상에서 파라미터값이 사진마다 다르게 적용되었다 그래서 일반화 성능이 아쉽다


### 한줄 요약 및 개인적 생각
감각적이나 직관적인 '스타일'이라는 개념을 수학적으로 정의하고 최적화가 가능하게끔 만들었다라는점에서
굉장히 참신하고 인상이 깊었다 첫 논문 리뷰인데 결과값이 한눈에 보일뿐더러 흥미로워 학부생 입장에서
딥러닝의 응용력을 느끼기에 좋은 논문이라 생각된다




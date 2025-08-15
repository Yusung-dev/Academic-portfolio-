# 🧠논문 구현: Image Style Transfer Using Convolutional Neural Networks

![result](../assets/result2.jpg)  

논문 링크: https://arxiv.org/abs/1508.06576

발표 학회/연도: CVPR 2016 (IEEE Conference on Computer Vision and Pattern Recognition)

논문 저자: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

------
### Overview

위 논문에서는 CNN을 이용하여 사진에서 content와 style을 추출해내 새로운 이미지를 생성하는 방식을 소개하였습니다

처음으로 논문을 구현해보았습니다 논문을 읽고 이해하며 그것을 코드로 옮기는데까지 많은 시간이 걸린 것 같습니다 첫번째로 구조를 나누는 것에 대해 고민했습니다
- `models.py` : 논문에 나온 아키텍처 구현  
- `loss.py` : style/content loss 구현  
- `train.py` : 전체 학습 루프

관습대로라면 'dataset.py'도 있어야하지만 이번 논문에서는 dataset이라고 하기엔 오로지 content사진1장, style사진 1장만이 필요했기에 포함하지 않았습니다  
추가로 이 논문을 review하기로 한 이유는 논문을 구현한뒤 결과물이 한 눈에 보이며 cnn을 이용하여 추상적의미인 style을 구현해냈다는 점이 매우 흥미로웠기에 선택하게 되었습니다

-----------

## Models.py

논문 2. Deep image representation을 살펴보면  
>**"The results presented below were generated on the ba
sis of the VGG network"**  
>-[Gatys et al., Image Style Transfer Using CNNs, CVPR 2016]

>**"We used the feature
 space provided by a normalised version of the 16 convo
lutional and 5 pooling layers of the 19-layer VGG network"**  
>-[Gatys et al., Image Style Transfer Using CNNs, CVPR 2016]

저자는 <strong>VGG19 model</strong>을 사용했다고 했습니다.
그중에서도 <strong>feature map</strong>을 중요하게 사용한 것을 알 수 있습니다.

>**"We reconstruct the input image from from layers ‘conv1 2’ (a),
 ‘conv2 2’ (b), ‘conv3 2’ (c), ‘conv4 2’ (d) and ‘conv5 2’ (e) of the original VGG-Network.k"**  
>-[Gatys et al., Image Style Transfer Using CNNs, CVPR 2016]

>**" We reconstruct
 the style of the input image from a style representation built on different subsets of CNN layers ( ‘conv1 1’ (a), ‘conv1 1’ and ‘conv2 1’
 (b), ‘conv1 1’, ‘conv2 1’ and ‘conv3 1’ (c), ‘conv1 1’, ‘conv2 1’, ‘conv3 1’ and ‘conv4 1’ (d), ‘conv1 1’, ‘conv2 1’, ‘conv3 1’, ‘conv4 1’
 and ‘conv5 1’ (e)"**  
>-[Gatys et al., Image Style Transfer Using CNNs, CVPR 2016]

어떤 피쳐맵을 사용했는지 궁금해질 무렵, 논문 속 위 문장에서 <code>conv(a)_(b)</code>와 같은 말이 나오는데, VGG19 구조를 참고하면 그 의미를 이해할 수 있습니다.

<div style="display: flex; align-items: flex-start;">
  <img src="../assets/paper3.jpg" width="300" style="margin-right: 20px;">
</div>

conv(a)_(b)는 저자가 특정 conv 레이어 위치를 명시하는 방식으로  
- `a` : 블록번호(VGG에서 max pooling으로 나누니 블록들)  
- `b` : 그 블록 안에서의 conv레이어 순서  

라는걸 알 수 있었습니다

그래서 저는 style사진에서 conv1_1, conv2_1, conv3_1, conv4_1, conv5_1을, content사진에서 conv4_2를 사용하였습니다 pytorch vgg19모델에서 다음과 같이 conv layer를 정리하였고 vgg19를 통과하다가 워원하는 conv를 만나게되면 해당 feature map을 가져올 수 있도록 코드를 제작하였습니다

```ptyhon
conv = {
    'conv1_1' : 0, #style featuremap layer 
    'conv2_1' : 5, #style featuremap layer 
    'conv3_1' : 10, #style featuremap layer 
    'conv4_1' : 19, #style featuremap layer 
    'conv5_1' : 28, #style featuremap layer 
    'conv4_2' : 21, #content featuremap layer
}
```

### Why CNN?

<div style="display: flex; align-items: flex-start;">
  <img src="../assets/paper4.jpg" width="400" style="margin-right: 20px;">
</div>
-[Gatys et al., Image Style Transfer Using CNNs, CVPR 2016]
<br><br>

그리고 이 사진과 함께 글에서 cnn을 사용한 이유를 들을 수 있었습니다  <br>

>**We find that reconstruction from lower layers is
 almost perfect (a–c). In higher layers of the network, detailed pixel information is lost while the high-level content of the image is preserved(d,e)**  
>-[Gatys et al., Image Style Transfer Using CNNs, CVPR 2016]

>**This creates images that match the style of a given image on an increasing scale while discarding information of the global arrangement of the scene.**  
>-[Gatys et al., Image Style Transfer Using CNNs, CVPR 2016]

content를 뽑기위해 cnn을 사용함으로써 high-level에서 세부적인 픽셀정보가 손실되지만 이미지의 고수준 내용을 보존해냈고  
style은 cnn을 사용함으로써 주어진 이미지의 스타일을 점점 더 큰 스케일에서 일치시키는 이미지를 만들어내, 장면의 전체 구성에 대한 정보를 제거함으로써 style을 뽑아낼 수 있었습니다  
이렇게 cnn이 content와 style을 뽑아냈기에 이러한 논문이 나올 수 있었지 않았나 생각됩니다
<br><br>

### 앞선 내용을 바탕으로 수도코드 만들기

```python
class StyleTransfer:

    초기화할 때:
        VGG19 모델 불러옴 (pretrained)
        VGG19에서 feature 추출 부분만 뽑음
        스타일 추출에 쓸 conv 레이어 번호들 정함 (예: conv1_1, conv2_1 ...)
        컨텐츠 추출에 쓸 conv 레이어 번호 정함 (예: conv4_2)

    forward 함수 (이미지, 모드):
        features라는 빈 리스트 만듦

        만약 모드가 'style'이면:
            VGG19 레이어를 차례대로 지나가면서
                현재 레이어 번호가 스타일 레이어 목록에 있으면
                    해당 레이어 출력값을 features에 추가

        만약 모드가 'content'이면:
            VGG19 레이어를 차례대로 지나가면서
                현재 레이어 번호가 컨텐츠 레이어 목록에 있으면
                    해당 레이어 출력값을 features에 추가

        최종적으로 features 반환
```
<br>

### model 만들기

```python
#import
import torch
import torch.nn as nn
from torchvision.models import vgg19

conv = {
    'conv1_1' : 0, #style featuremap layer 
    'conv2_1' : 5, #style featuremap layer 
    'conv3_1' : 10, #style featuremap layer 
    'conv4_1' : 19, #style featuremap layer 
    'conv5_1' : 28, #style featuremap layer 
    'conv4_2' : 21, #content featuremap layer
}

class StyleTransfer(nn.Module):
    def __init__(self,):
        super(StyleTransfer, self).__init__()
        #TODO: VGG19 load
        self.vgg19_model = vgg19(pretrained = True)
        self.vgg19_features = self.vgg19_model.features

        #TODO: conv layer 분리
        self.style_layer = [conv['conv1_1'], conv['conv2_1'], conv['conv3_1'], conv['conv4_1'], conv['conv5_1']]
        self.content_layer = [conv['conv4_2']]

        pass

    def forward(self, x, mode:str):
        #TODO : style, content마다 conv layer slicing해서 사용하기
        features = []

        if mode == 'style':
            for i in range(len(self.vgg19_features)):
                x = self.vgg19_features[i](x)
                if i in self.style_layer:
                    features.append(x)

        if mode == 'content':
            for i in range(len(self.vgg19_features)):
                x = self.vgg19_features[i](x)
                if i in self.content_layer:
                    features.append(x)

        return features
```

----------

## loss.py

다음으로는 loss를 구현했습니다<br>

논문에 각 content loss, style loss가 나와있었고 그 두개를 합친 total_loss까지 잘 설명되어있었습니다<br><br>

### content loss
```math
\mathcal{L}_{\mathrm{content}}(\vec{p}, \vec{x}, l)
= \frac{1}{2} \sum_{i,j} \left( F^l_{ij} - P^l_{ij} \right)^{2}
```
```math
\frac{\partial \mathcal{L}_{\mathrm{content}}}{\partial F^l_{ij}} =
\begin{cases}
F^l_{ij} - P^l_{ij}, & \text{if } F^l_{ij} > 0, \\
0, & \text{if } F^l_{ij} < 0
\end{cases}
```
<br>
  
- $`\vec{x}`$ : 생성된 입력 이미지  
- $`\vec{p}`$ : 원본 이미지  
- $`\mathcal{l}`$ : 이미지 층  
- $`\mathcal{P}(l)`$, $`\mathcal{F}(l)`$ : 원본 이미지와 생성된 입력 이미지 각 이미지 $`\mathcal{l}`$ 층에서의 특징 표현  

임을 논문에서 찾을 수 있었습니다그럼으로써 MSE를 구하고 그것으로 backpropagation을 계산해서 무작위 이미지인 $`\vec{x}`$ 를 점차적으로 수정해 CNN의 특정 층에서 원래 이미지 $`\vec{p}`$ 와 동일한 반응을 생성하도록 만들어 입력이미지인 $`\vec{x}`$ 는 네트워크의 처리 계층을 따라 실제 내용인 content에 더 민감해지는 표현으로 변환되지만 그 정확한 외형은 불변해집니다 즉, 네트워크의 높은 층들은 입력 이미지 내 객체와 그 배열 측면에서 고수준의 내용을 포착하지만 복원 시의 정확한 픽셀 값에 대해서는 큰 제약을 두지 않고 이에 반해, 낮은 층들로부터의 복원은 원래 이미지의 정확한 픽셀 값을 단순히 재현합니다<br><br><br>

### style loss

```math
G^l_{ij} = \sum_k F^l_{ik} F^l_{jk} \\[1em]
```
```math
E_l = \frac{1}{4 N_l^2 M_l^2} \sum_{i,j} \left( G^l_{ij} - A^l_{ij} \right)^{2} \\[1em]
```
```math
\mathcal{L}_{\mathrm{style}}(\vec{a}, \vec{x}) = \sum_{l=0}^{L} w_l E_l \\[1em]
```
```math
\frac{\partial E_l}{\partial F^l_{ij}} =
\begin{cases}
\frac{1}{N_l^2 M_l^2} \left( (F^l)^{\mathrm{T}} \left( G^l - A^l \right) \right)_{ji}, & \text{if } F^l_{ij} > 0, \\
0, & \text{if } F^l_{ij} < 0
\end{cases}
```
<br>

- $`\mathcal{G}(i,j)`$ : i와 j 간의 내적
- $`\vec{x}`$ : 원본 이미지  
- $`\mathcal{A}(l)`$, $`\mathcal{F}(l)`$ : 원본 이미지 $`\mathcal{l}`$ 층에서의 특징 표현  
- $`\mathcal{w}`$ : 각 층이 전체 손실에 기여하는 정도를 조절하는 가중치 계수  

임을 논문에서 찾을 수 있었습니다 입력 이미지의 스타일에 대한 표현을 얻기 위해 질감 정보를 포착하도록 설계된 특징 공간을 사용합니다 특징 상관관계는 $`\mathcal{G}(l)`$ 이 만들러입니다 스타일 표현에 포함된 각 층마다 $`\mathcal{G}(l)`$ 과 $`\mathcal{A}(l)`$ 간의 요소별 평균 제곱오차가 계산되어 스타일 손실 $`\mathcal{L}_{\text{style}}`$ 이 됩니다<br><br>

### total loss

```math
\mathcal{L}_{\mathrm{total}}(\vec{p}, \vec{a}, \vec{x})
= \alpha\, \mathcal{L}_{\mathrm{content}}(\vec{p}, \vec{x})
+ \beta\, \mathcal{L}_{\mathrm{style}}(\vec{a}, \vec{x})
```
<br>

 - '&alpha;, &beta;' : 하이퍼 파라미터

총 손실 $`\mathcal{L}_{\text{total}}`$ 은 내용 손실과 스타일 손실의 선형 결합니였습니다 이 총손실을 이미지의 픽셀 값들에 대해 미분한 값은 오차 역전파로 계산될 수 있었으며 이 그래디언트가 이미 $`\vec{x}`$ 를 반복적으로 갱신하는데 사용되고 결국 스타일 $`\vec{a}`$ 의 스타일 특징과 내용 이미지 $`\vec{p}`$ 의 내용 특징을 동시에 일치시킬 수 있었습니다

추가적으로 논문 result부분에서

>**The ratio α/β was either 1 × 10^−3 (Fig 3 B), 8 × 10^−4 (Fig 3
 C), 5 ×10^−3 (Fig 3 D), or 5×10^−4 (Fig 3 E,F).  
>-[Gatys et al., Image Style Transfer Using CNNs, CVPR 2016]

각 사진마다 하이퍼 파라미터값을 다르게 설정하였다는 것 또한 얻어낼 수 있었습니다
<br><br>

### loss 만들기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

#contentloss정의
class ContentLoss(nn.Module):
    def __init__(self,):
        super(ContentLoss, self).__init__()

    def forward(self, x, y):
        loss = F.mse_loss(x,y)
        return loss

#styleloss정의
class StyleLoss(nn.Module):
    def __init__(self,):
        super(StyleLoss, self).__init__()

    def gram_matrix(self, x):
        b,c,h,w = x.size()
        featrues = x.view(b,c,h*w)
        features_T = featrues.transpose(1,2)
        G = torch.matmul(featrues, features_T)

        return G.div(b*c*h*w)

    def forward(self, x, y):

        Gx = self.gram_matrix(x)
        Gy = self.gram_matrix(y)
        loss = F.mse_loss(x, y)
        return loss
```
----------

## train.py

마지막으로 train.py를 구현해 보았습니다
<br>

1. preprocessing
    - 입력 이미지를 512×512로 리사이즈하고, ToTensor()로 float32 텐서로 바꿉니다. 결과 형태는 (1, C, H, W)입니다.
    - ImageNet 통계(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])로 정규화하여 VGG 기반 특성 추출에 맞춥니다.
    - 최종 반환은 배치 차원을 붙인 텐서로, 바로 모델에 넣을 수 있는 형태입니다.

2. postprocessing
    - 학습 중인 생성 이미지 텐서를 CPU로 옮기고 numpy로 변환한 뒤, (H,W,C)로 전치합니다.
    - 정규화의 역변환(denorm)을 적용해 원래 픽셀 스케일로 되돌립니다: image = image*std + mean 후 [0,1]로 클리핑하고 uint8로 변환합니다.
    - PIL.Image로 다시 바꿔서 파일로 저장할 수 있도록 합니다.

3. 모델/손실 구성
    - StyleTransfer()는 백본에서 중간 feature를 뽑아, 모드에 따라 두 가지 표현을 제공합니다.
        - `content`: 콘텐츠 손실 계산용 feature 리스트
        - `style`: 스타일 손실 계산용 feature 리스트(보통 Gram matrix 계산에 적합한 레이어들)
    - ContentLoss, StyleLoss는 각각 콘텐츠 일치, 스타일 일치를 측정합니다. 코드 본문에서는 각 레이어별 손실을 합산합니다.

4. 하이퍼파라미터
    - `alpha=1` : 콘텐츠 손실 가중치
    - `beta=1e6`: 스타일 손실 가중치
    - `lr=1` : 픽셀을 직접 최적화하므로 비교적 큰 학습률을 사용

5. 디바이스 설정
    - GPU가 있으면 cuda, 없으면 cpu에서 실행합니다. 모델과 입력 텐서를 같은 디바이스로 옮겨 일관성 있게 연산합니다.

6. 초기 이미지 설정
    - 논문상에서는 white image를 사용하였지만 조금 더 좋은 결과물을 얻기위해 이미지 그자체(content image)를 사용하였습니다.
    - x.requires_grad_(True)로 픽셀을 미분 가능하게 만들어 옵티마이저가 업데이트할 수 있게 합니다.

7. 옵티마이저
    - optim.Adam([x], lr=lr)로 단 하나의 파라미터 집합(이미지 텐서 x)만 최적화합니다.

8. 전체 파이프라인 요약
   1. 콘텐츠/스타일 이미지를 로드하고 동일한 전처리를 적용
   2. 미리 학습된 특성 추출 네트워크로 콘텐츠/스타일 타깃 표현 고정
   3. 초기 이미지 x를 변수로 두고, 콘텐츠/스타일 손실을 최소화하도록 x를 직접 최적화
   4. 주기적으로 중간 결과를 복원해 저장하며 수렴 상황을 확인
<br>

```python
#import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os

import numpy as np
from PIL import Image

from models import StyleTransfer
from loss import ContentLoss, StyleLoss
from tqdm import tqdm


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def pre_processing(image:Image.Image) -> torch.Tensor:
    preprocessing = T.Compose([
        T.Resize((512,512)), #이미지 resize
        T.ToTensor(), #image to tensor
        T.Normalize(mean, std) # lambda x : (x-mean) / std
    ]) # (c, h ,w)

    # (1, c, h ,w)
    image_tensor:torch.Tensor = preprocessing(image)

    return image_tensor.unsqueeze(0)

def post_processing(tensor:torch.Tensor) -> Image.Image:

    # shape 1,c,h,w
    image:np.ndarray = tensor.to('cpu').detach().numpy()
    # shape c,h,w
    image = image.squeeze()
    # shape h,w,c
    image = image.transpose(1, 2, 0)
    # de norm
    image = image*std + mean
    # clip
    image = image.clip(0,1)*255
    # dtype uint8
    image = image.astype(np.uint8)
    # numpy -> Image
    return Image.fromarray(image)


def train_main():
    # load data
    content_image = Image.open('./content.jpg')
    content_image = pre_processing(content_image)

    style_image = Image.open('./style.jpg')
    style_image = pre_processing(style_image)

    # load model
    style_transfer = StyleTransfer().eval()

    # load loss
    content_loss = ContentLoss()
    style_loss = StyleLoss()

    # hyper parameter
    alpha = 1
    beta = 1e6
    lr = 1

    save_root = f'{alpha}_{beta}_{lr}_style_transfer_01'
    os.makedirs(save_root, exist_ok=True)

    # device setting
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'


    style_transfer = style_transfer.to(device)
    content_image = content_image.to(device)
    style_image = style_image.to(device)
    
    # noise
    # x = torch.randn(1,3,512,512).to(device) 노이즈 시작사진
    x = content_image.clone()                #고양이 시작사진
    x.requires_grad_(True)

    # setting optimizer
    optimizer = optim.Adam([x], lr=lr)

    # train loop
    steps = 500
    for step in tqdm(range(steps)):
        ## content representation (x, content_image)
        ## style representation (x, style_image)

        x_content_list = style_transfer(x, 'content')
        y_content_list = style_transfer(content_image, 'content')

        x_style_list = style_transfer(x, 'style')
        y_style_list = style_transfer(style_image, 'style')

        ## loss_content, loss_style
        loss_c = 0
        loss_s = 0
        loss_total = 0

        for x_content, y_content in zip(x_content_list, y_content_list):
            loss_c += content_loss(x_content, y_content)
        loss_c = alpha*loss_c

        for x_style, y_style in zip(x_style_list, y_style_list):
            loss_s += style_loss(x_style, y_style)
        loss_s = beta*loss_s

        loss_total = loss_c + loss_s

        ## optimizer step
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        ## loss print
        if step%100==0:
            print(f"loss_c: {loss_c.cpu()}")
            print(f"loss_s: {loss_s.cpu()}")
            print(f"loss_total: {loss_total.cpu()}")

            ## post processing
            ## image gen output save
            gen_img:Image.Image = post_processing(x)
            gen_img.save(os.path.join(save_root, f'{step}.jpg'))

if __name__=="__main__":
    train_main()
```

--------

## 수행결과

수행은 code는 colab에서 실행 진행하였습니다
<br>

![result](../assets/result3.jpg)  

다음사진은 제가 구현한 실제 코드로 만든 사진입니다 content 이미지는 귀여운 고양이를 사용하였으며 style이미지로 다양한 패턴과 색생이 가미되어있는 패턴이미지를 사용하였습니다 두 사진의 결과물 다 100epoch째 사진이며 결과가 매우 잘 나온 것을 볼 수 있었습니다
<br>

<div style="display: flex; align-items: flex-start;">
  <img src="../assets/result4.jpg" width="400" style="margin-right: 20px;">
</div>
<br>
실제 loss값을 확인해보면 100epoch만에 total loss가 수렴한걸 확인 할 수 있었습니다

---------
### 소감

논문을 읽고 이렇게 코딩까지하는 과정이 처음해보면서 예상보다 훨씬 많은 시간과 노력이 필요하다는 것을 느꼈습니다 논문 속 수식과 알고리즘을 이해하기 위해 여러 번 읽고 참고 자료를 찾아보았으며, 이를 코드로 옮기는 과정에서 시행착오도 많았습니다 특히 논문과 실제 구현간의 세부 차이를 맞추는 일이 제일 힘들었고 그 과정에서 모델 구조와 동작 원리를 깊이 이해할 수 있었던 것 같습니다 힘들었지만 이 경험을 통해 단순한 읽기에 그치지 않고, 연구 내용을 응용해보고, 더 많은 논문을 읽어 앞으로 공부와 연구에 큰 자산이 될 수 있도록 하겠습니다

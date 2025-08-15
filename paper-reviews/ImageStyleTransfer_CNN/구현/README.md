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
<div style="display: flex; align-items: flex-start;">
  <img src="../assets/cal1.jpg" width="300" style="margin-right: 20px;">
  <br>
  <img src="../assets/cal2.jpg" width="300" style="margin-right: 20px;">
</div>
  
- `x` : 생성된 입력 이미지  
- `p` : 원본 이미지  
- `l` : 이미지 층  
- `P(l), F(l)` : 원본 이미지와 생성된 입력 이미지 각 이미지 l층에서의 특징 표현  

임을 논문에서 찾을 수 있었습니다그럼으로써 mse를 구하고 그것으로 backpropagation을 계산해서 무작위 이미지인 x를 점차적으로 수정해 cnn의 특정 층에서 원래 이미지 p와 동일한 반응을 생성하도록 만들어 입력이미지인 x는 네트워크의 처리 계층을 따라 실제 내용인 content에 더 민감해지는 표현으로 변환되지만 그 정확한 외형은 불변해집니다 즉, 네트워크의 높은 층들은 입력 이미지 내 객체와 그 배열 측면에서 고수준의 내용을 포착하지만 복원 시의 정확한 픽셀 값에 대해서는 큰 제약을 두지 않고 이에 반해, 낮은 층들로부터의 복원은 원래 이미지의 정확한 픽셀 값을 단순히 재현합니다<br><br><br>

### style loss
<div style="display: flex; align-items: flex-start;">
  <img src="../assets/cal3.jpg" width="150" style="margin-right: 20px;">
  <br>
  <img src="../assets/cal4.jpg" width="200" style="margin-right: 20px;">
  <br>
  <img src="../assets/cal5.jpg" width="200" style="margin-right: 20px;">
  <br>
  <img src="../assets/cal6.jpg" width="300" style="margin-right: 20px;">
</div>
<br>
  
- `G(i,j)` : i와 j 간의 내적
- `a` : 원본 이미지  
- `A(l), F(l)` : 원본 이미지 l층에서의 특징 표현  
- `w` : 각 층이 전체 손실에 기여하는 정도를 조절하는 가중치 계수  

임을 논문에서 찾을 수 있었습니다 입력 이미지의 스타일에 대한 표현을 얻기 위해 질감 정보를 포착하도록 설계된 특징 공간을 사용합니다 특징 상관관계는 G(l)이 만들러입니다 스타일 표현에 포함된 각 층마다 G(l)과 A(l)간의 요소별 평균 제곱오차가 계산되어 스타일 손실 L(style)이 됩니다<br><br><br>

### total loss
<div style="display: flex; align-items: flex-start;">
  <img src="../assets/cal7.jpg" width="300" style="margin-right: 20px;">
 <br>

 - '&alpha;, &beta;' : 하이퍼 파라미터

총 손실 L(total)은 내용 손실과 스타일 손실의 선형 결합니였습니다 이 총손실을 이미지의 픽셀 값들에 대해 미분한 값은 오차 역전파로 계산될 수 있었으며 이 그래디언트가 이미 x를 반복적으로 갱신하는데 사용되고 결국 스타일 a의 스타일 특징과 내용 이미지 p의 내용 특징을 동시에 일치시킬 수 있었습니다

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

- 수행은 code는 colab에서 실행 진행하였습니다


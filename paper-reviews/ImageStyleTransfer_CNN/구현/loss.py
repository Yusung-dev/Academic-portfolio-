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
        

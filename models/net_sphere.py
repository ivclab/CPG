# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torchvision.models as models

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.mlambda = [
            lambda x: x**0, # cos(0*theta)=1
            lambda x: x**1, # cos(1*theta)=cos(theta)
            lambda x: 2*x**2-1, # cos(2*theta)=2*cos(theta)**2-1
            lambda x: 4*x**3-3*x, # cos(3*theta)=4*cos(theta)**3-3cos(theta)
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        # input为输入的特征，(B, C)，B为batchsize，C为图像的类别总数
        x = input   # size=(B,F)，F为特征长度，如512
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        # 对w进行归一化，renorm使用L2范数对第1维度进行归一化，将大于1e-5的截断，乘以1e5，
        # 使得最终归一化到1.如果1e-5设置的过大，裁剪时某些很小的值最终可能小于1
        # 注意，第0维度只对每一行进行归一化（每行平方和为1），
        # 第1维度指对每一列进行归一化。由于w的每一列为x的权重，因而此处需要对每一列进行归一化。
        # 如果要对x归一化，需要对每一行进行归一化，此时第二个参数应为0
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        # 对输入x求平方，而后对不同列求和，再开方，得到每行的模，最终大小为第0维的，即B
        # (由于对x不归一化，但是计算余弦时需要归一化，因而可以先计算模。
        # 但是对于w，不太懂为何不直接使用这种方式，而是使用renorm函数？)
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum
        # 对权重w求平方，而后对不同行求和，再开方，得到每列的模
        # （理论上之前已经归一化，此处应该是1，但第一次运行到此处时，并不是1，不太懂），最终大小为第1维的，即C

        cos_theta = x.mm(ww) # size=(B,Classnum)
        # 矩阵相乘(B,F)*(F,C)=(B,C)，得到cos值，由于此处只是乘加，故未归一化
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        # 对每个cos值均除以B和C，得到归一化后的cos值
        cos_theta = cos_theta.clamp(-1,1)
        # 将cos值截断到[-1,1]之间，理论上不截断应该也没有问题，毕竟w和x都归一化后，cos值不可能超出该范围
        # ------------------------------------------------
        cos_m_theta = self.mlambda[self.m](cos_theta)
        # 通过cos_theta计算cos_m_theta，mlambda为cos_m_theta展开的结果
        theta = Variable(cos_theta.data.acos())
        # 通过反余弦，计算角度theta，(B,C)
        k = (self.m*theta/3.14159265).floor()
        # 通过公式，计算k，(B,C)。此处为了保证theta大于k*pi/m，转换过来就是m*theta/pi，再向上取整
        n_one = k*0.0 - 1
        # 通过k的大小，得到同样大小的-1矩阵，(B,C)
        phi_theta = (n_one**k) * cos_m_theta - 2*k
        # 通过论文中公式，得到phi_theta。(B,C)
        # --------------------------------------------
        cos_theta = cos_theta * xlen.view(-1,1)
        # 由于实际上不对x进行归一化，此处cos_theta需要乘以B。(B,C)
        phi_theta = phi_theta * xlen.view(-1,1)
        # 由于实际上不对x进行归一化，此处phi_theta需要乘以B。(B,C)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input # cos_theta，(B,C)。 phi_theta，(B,C)
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        # 得到和cos_theta相同大小的全0矩阵。(B,C)
        index.scatter_(1,target.data.view(-1,1),1)
        # 得到一个one-hot矩阵，第i行只有target[i]的值为1，其他均为0
        index = index.byte()# index为float的，转换成byte类型
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it )) # 得到lamb
        output = cos_theta * 1.0 #size=(B,Classnum)
        # 如果直接使用output=cos_theta，可能不收敛(未测试，但其他程序中碰到过直接对输入使用[index]无法收敛，加上*1.0可以收敛的情况)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)# 此行及下一行将target[i]的值通过公式得到最终输出
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output,dim=1) # 得到概率, ..I change this line (dim=1)
        logpt = logpt.gather(1,target) # 下面为交叉熵的计算（和focal loss的计算有点类似，当gamma为0时，为交叉熵）。
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp()) # ln(e) = 1

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        # target = target.view(-1)  # 若要简化，理论上可直接使用这两行计算交叉熵(此处未测试，在其他程序中使用后可以正常训练)
        # loss = F.cross_entropy(cos_theta, target)

        return loss

class sphere(nn.Module):
    def __init__(self,embedding_size,classnum,feature=False):
        super(sphere, self).__init__()
        self.embedding_size = embedding_size
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*112
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*56
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*28
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*28
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*14
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*14
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*14
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*14
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*7
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*7,self.embedding_size)
        self.fc6 = AngleLinear(self.embedding_size,self.classnum)


    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output


    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)
        #x = self.l2_norm(x)
        if self.feature:
            return x

        x = self.fc6(x)
        return x

class sphereVGG(nn.Module):
    def __init__(self,embedding_size,classnum,feature=False):
        super(sphereVGG, self).__init__()
        self.embedding_size = embedding_size
        self.classnum = classnum
        self.feature = feature
        # load feature extractor from vgg16_bn pretrained-model
        #self.vgg16_bn_feat_extractor = models.vgg16_bn(pretrained=False).features
        self.vgg16_bn_feat_extractor = nn.Sequential(*list(models.vgg16_bn(pretrained=False).features))
        # concatenate the embedding layer
        self.fc5 = nn.Linear(512*5*5,self.embedding_size)
        #self.fc6 = AngleLinear(self.embedding_size,self.classnum)
        self.fc6 = nn.Linear(self.embedding_size,self.classnum)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.vgg16_bn_feat_extractor(x)
        x = x.view(x.size(0),-1)
        x = self.fc5(x)
        #x = self.l2_norm(x)
        if self.feature:
            return x
        x = self.fc6(x)
        return x
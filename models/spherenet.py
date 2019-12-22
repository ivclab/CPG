import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import models.layers as nl
import pdb
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

__all__ = ['spherenet20', 'AngleLoss']


class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


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
        cos_theta,phi_theta = input
        target = target.view(-1,1)

        index = cos_theta.data * 0.0
        index.scatter_(1,target.data.view(-1,1),1)
        # index = index.byte()
        index = index.float()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))

        # output = cos_theta * 1.0
        # output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        # output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        output = cos_theta * 1.0
        output -= cos_theta * index *(1.0+0)/(1+self.lamb)
        output += phi_theta * index *(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()
        return loss


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
        x = input
        w = self.weight
        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)
        wlen = ww.pow(2).sum(0).pow(0.5)
        cos_theta = x.mm(ww)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = Variable(cos_theta.data.acos())
        k = (self.m*theta/3.14159265).floor()
        n_one = k*0.0 - 1
        phi_theta = (n_one**k) * cos_m_theta - 2*k
        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output


class SphereNet(nn.Module):
    def __init__(self, dataset_history, dataset2num_classes, network_width_multiplier=1.0, shared_layer_info={}, init_weights=True):
        super(SphereNet, self).__init__()
        self.network_width_multiplier = network_width_multiplier
        self.make_feature_layers()

        self.shared_layer_info = shared_layer_info
        self.datasets = dataset_history
        self.classifiers = nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes

        if self.datasets:
            self._reconstruct_classifiers()

        if init_weights:
            self._initialize_weights()
        return

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
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def forward_to_embeddings(self, x):
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
        x = self.flatten(x)
        x = self.classifier[0](x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nl.SharableConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
        return

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            if 'face_verification' in dataset:
                embedding_size = 512
                classifier_module = nn.Sequential(nn.Linear(int(self.shared_layer_info[dataset]['network_width_multiplier']*512)*7*7,
                                                            embedding_size),
                                                  AngleLinear(embedding_size, num_classes))
                self.classifiers.append(classifier_module)
            else:
                self.classifiers.append(nn.Linear(int(self.shared_layer_info[dataset]['network_width_multiplier']*512)*7*7,
                                                  num_classes))
        return

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            if 'face_verification' in dataset:
                embedding_size = 512
                classifier_module = nn.Sequential(nn.Linear(int(self.network_width_multiplier*512)*7*7, embedding_size),
                                                  AngleLinear(embedding_size, num_classes))
                self.classifiers.append(classifier_module)
                nn.init.normal_(classifier_module[0].weight, 0, 0.01)
                nn.init.constant_(classifier_module[0].bias, 0)
                nn.init.normal_(classifier_module[1].weight, 0, 0.01)
            else:
                self.classifiers.append(nn.Linear(int(self.network_width_multiplier*512)*7*7, num_classes))
                nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
                nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)
        return

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]
        return

    def make_feature_layers(self):
        ext = self.network_width_multiplier
        self.conv1_1 = nl.SharableConv2d(3,int(64*ext),3,2,1) #=>B*int(64*ext)*56*56
        self.relu1_1 = nn.PReLU(int(64*ext))
        self.conv1_2 = nl.SharableConv2d(int(64*ext), int(64*ext),3,1,1)
        self.relu1_2 = nn.PReLU(int(64*ext))
        self.conv1_3 = nl.SharableConv2d(int(64*ext), int(64*ext),3,1,1)
        self.relu1_3 = nn.PReLU(int(64*ext))

        self.conv2_1 = nl.SharableConv2d(int(64*ext), int(128*ext),3,2,1) #=>B*int(128*ext)*28*28
        self.relu2_1 = nn.PReLU(int(128*ext))
        self.conv2_2 = nl.SharableConv2d(int(128*ext),int(128*ext),3,1,1)
        self.relu2_2 = nn.PReLU(int(128*ext))
        self.conv2_3 = nl.SharableConv2d(int(128*ext),int(128*ext),3,1,1)
        self.relu2_3 = nn.PReLU(int(128*ext))

        self.conv2_4 = nl.SharableConv2d(int(128*ext),int(128*ext),3,1,1) #=>B*int(128*ext)*28*28
        self.relu2_4 = nn.PReLU(int(128*ext))
        self.conv2_5 = nl.SharableConv2d(int(128*ext),int(128*ext),3,1,1)
        self.relu2_5 = nn.PReLU(int(128*ext))


        self.conv3_1 = nl.SharableConv2d(int(128*ext),int(256*ext),3,2,1) #=>B*int(256*ext)*14*14
        self.relu3_1 = nn.PReLU(int(256*ext))
        self.conv3_2 = nl.SharableConv2d(int(256*ext),int(256*ext),3,1,1)
        self.relu3_2 = nn.PReLU(int(256*ext))
        self.conv3_3 = nl.SharableConv2d(int(256*ext),int(256*ext),3,1,1)
        self.relu3_3 = nn.PReLU(int(256*ext))

        self.conv3_4 = nl.SharableConv2d(int(256*ext),int(256*ext),3,1,1) #=>B*int(256*ext)*14*14
        self.relu3_4 = nn.PReLU(int(256*ext))
        self.conv3_5 = nl.SharableConv2d(int(256*ext),int(256*ext),3,1,1)
        self.relu3_5 = nn.PReLU(int(256*ext))

        self.conv3_6 = nl.SharableConv2d(int(256*ext),int(256*ext),3,1,1) #=>B*int(256*ext)*14*14
        self.relu3_6 = nn.PReLU(int(256*ext))
        self.conv3_7 = nl.SharableConv2d(int(256*ext),int(256*ext),3,1,1)
        self.relu3_7 = nn.PReLU(int(256*ext))

        self.conv3_8 = nl.SharableConv2d(int(256*ext),int(256*ext),3,1,1) #=>B*int(256*ext)*14*14
        self.relu3_8 = nn.PReLU(int(256*ext))
        self.conv3_9 = nl.SharableConv2d(int(256*ext),int(256*ext),3,1,1)
        self.relu3_9 = nn.PReLU(int(256*ext))
        self.conv4_1 = nl.SharableConv2d(int(256*ext),int(512*ext),3,2,1) #=>B*int(512*ext)*7*7
        self.relu4_1 = nn.PReLU(int(512*ext))
        self.conv4_2 = nl.SharableConv2d(int(512*ext),int(512*ext),3,1,1)
        self.relu4_2 = nn.PReLU(int(512*ext))
        self.conv4_3 = nl.SharableConv2d(int(512*ext),int(512*ext),3,1,1)
        self.relu4_3 = nn.PReLU(int(512*ext))
        self.flatten = View(-1, int(ext*512)*7*7)
        return


def spherenet20(dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, shared_layer_info={}, **kwargs):
    return SphereNet(dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)

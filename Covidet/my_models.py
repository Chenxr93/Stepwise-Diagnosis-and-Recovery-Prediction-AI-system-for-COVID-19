import sys
package_path = '/home/gaozebin/Project/New_ODIR/D2_code/EfficientNet-PyTorch/'
sys.path.append(package_path)


from lip_resnet import resnet50
from myres import resnet18
from myres import vgg16_bn
from shake_pyramidnet import ShakePyramidNet
from my_resnext import resnext50_32x4d

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.parameter import Parameter
import pretrainedmodels
from torch.autograd import Variable
from resnext_ws import l_resnext50

class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=64, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        #print(C)
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=32, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
model_weight_dir = {
        # 'resnet18': '/home/gaozebin/.cache/torch/checkpoints/resnet18-5c106cde.pth',
        'resnet18': '/data/chenxiangru/ODIR/awesome_ODIR/models/0/0_0.8521163658775249_0.9234104046242775.pkl',
        'seresnext50': '',
        'resnet50': '/data/chenxiangru/ncov/resnet50-19c8e357.pth',
        'resnext50': '/data/chenxiangru/ODIR/awesome_ODIR/tmp/resnext50_32x4d-7cdf4587.pth',
        'vgg16': '/home/gaozebin/.cache/torch/checkpoints/vgg16_bn-6c64b313.pth',
        'resnext_ws50':'/data/chenxiangru/covid2019/models/X-50-GN-WS.pth.tar'
        }

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


def init_with_pretrained(model, model_name):

    pre_dict = torch.load(model_weight_dir[model_name])
    #print("pre train:", list(pre_dict.keys())[:10])
    model_dict = model.state_dict()
    # for k in model_dict:
    #     print('model_dict:',k)

    #print("model:", list(model_dict.keys())[:10])
    for k in model_dict:
        # if 'fc' in k or 'classifier' in k or '_fc' in k or 'features.0' in k:
        #     continue
        if 'avgpool' in k:
            continue
        print(k)
        model_dict[k] = pre_dict['module.'+k]
    model.load_state_dict(model_dict)
    print("Loaded pretrained {} model from {}.".format(
        model_name, model_weight_dir[model_name]))
    return model


class AttributeDecoder(nn.Module):
    def __init__(self):
        super(AttributeDecoder, self).__init__()
        self.linear = nn.Linear(2000, 1000)
        # self.Batchnorm1D = nn.BatchNorm1d(1000)
        self.relu = nn.ReLU()
        self.linearout = nn.Linear(1000,1)

    def forward(self, x):
        x = self.linear(x)
        # x = self.Batchnorm1D(x)
        x = self.relu(x)
        x = self.linearout(x)
        out = torch.sigmoid(x)
        return out
class Res18X2WithoutDecoder(nn.Module):
    def __init__(self):
        super(Res18X2WithoutDecoder, self).__init__()
        self.v1 = models.resnext50_32x4d()  # models.resnet18()
        self.v2 = models.resnext50_32x4d()  # models.resnet18()
        #print(self.v1)
        #for i in self.v1.named_modules():
            # if 'bn' in i[0]:
            #     name = i[0].split('.')
            #     m = self.v1
            #     for p in range(len(name) - 1):
            #         m = getattr(m, name[p])
            #     bn = getattr(m, name[-1])
            #     num_features = bn.num_features
            #     setattr(m, name[-1], GroupNorm(num_features))
            # if 'conv' in i[0]:
            #     name = i[0].split('.')
            #     m = self.v2
            #     for p in range(len(name) - 1):
            #         m = getattr(m, name[p])
            #     conv = getattr(m, name[-1])
            #     setattr(m, name[-1], Conv2d(*[conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
            #      conv.padding, conv.dilation, conv.groups, conv.bias]))
            #     print('conv.groups',conv.groups)
        #for i in self.v2.named_modules():
            # if 'bn' in i[0]:
            #     name = i[0].split('.')
            #     m = self.v2
            #     for p in range(len(name) - 1):
            #         m = getattr(m, name[p])
            #     bn = getattr(m, name[-1])
            #     num_features = bn.num_features
            #     setattr(m, name[-1], GroupNorm(num_features))
            # if 'Conv2d' in i[0]:
            #     name = i[0].split('.')
            #     m = self.v2
            #     for p in range(len(name) - 1):
            #         m = getattr(m, name[p])
            #     conv = getattr(m, name[-1])
            #     setattr(m, name[-1], Conv2d(*[conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
            #      conv.padding, conv.dilation, conv.groups, conv.bias]))
        # self.v1 = ShakePyramidNet(depth=50, alpha=30, label=1000)
        # self.v2 = ShakePyramidNet(depth=50, alpha=30, label=1000)
        #self.v1 = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
        # self.v1.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        #self.v1.avg_pool = GeM()
        #self.v2 = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
        # self.v2.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        #self.v2.avg_pool = GeM()
    def forward(self, x):
        x1 = x[:, :3, ...]
        x2 = x[:, 3:, ...]
        x1 = self.v1(x1)
        x2 = self.v2(x2)
        out = torch.cat([x1, x2], dim=1)
        return out
class Res18X2(nn.Module):
    def __init__(self, num_classes=2):
        super(Res18X2, self).__init__()
        # self.v1 = resnet18() # models.resnet18()
        # self.v2 = resnet18() # models.resnet18()

        self.v1 = models.resnext50_32x4d()  # models.resnet18()
        self.v2 = models.resnext50_32x4d()  # models.resnet18()

        # self.v1 = ShakePyramidNet(depth=50, alpha=30, label=1000)
        # self.v2 = ShakePyramidNet(depth=50, alpha=30, label=1000)

        #self.v1 = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
        # self.v1.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        #self.v1.avg_pool = GeM()

        #self.v2 = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
        # self.v2.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        #self.v2.avg_pool = GeM()

        self.c3 = nn.Sequential(
                nn.Linear(2000, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, num_classes)
                )


    def forward(self, x):
        x1 = x[:, :3, ...]
        x2 = x[:, 3:, ...]

        x1 = self.v1(x1)
        x2 = self.v2(x2)

        out = torch.cat([x1, x2], dim=1)
        out = self.c3(out)
        return out

class Vgg16X2(nn.Module):
    def __init__(self, num_classes=2):
        super(Vgg16X2, self).__init__()
        self.v1 = vgg16_bn()
        self.v2 = vgg16_bn()

        self.c3 = nn.Sequential(
                nn.Linear(2000, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 2)
                )


    def forward(self, x):
        x1 = x[:, :3, ...]
        x2 = x[:, 3:, ...]

        x1 = self.v1(x1)
        x2 = self.v2(x2)

        out = torch.cat([x1, x2], dim=1)
        out = self.c3(out)
        return out
class Resnext50X2(nn.Module):
    def __init__(self, num_classes=8):
        super(Resnext50X2, self).__init__()
        model_name = 'se_resnext50_32x4d'  # could be fbresnet152 or inceptionresnetv2
        self.v1 = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        dim_feats = self.v1.last_linear.in_features  # =2048
        self.v1.avg_pool = nn.Sequential(nn.AvgPool2d(kernel_size=16, stride=1, padding=0),
                                       #nn.Conv2d(dim_feats,  dim_feats, kernel_size=2, stride=1, padding=0),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(dim_feats))
        self.v1.last_linear = nn.Linear(dim_feats, 1000)
        self.v2 = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        dim_feats = self.v2.last_linear.in_features  # =2048
        self.v2.avg_pool = nn.Sequential(nn.MaxPool2d(kernel_size=16, stride=1, padding=0),
                                         #nn.Conv2d(dim_feats, 2 * dim_feats, kernel_size=2, stride=1, padding=0),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(dim_feats))
        self.v2.last_linear = nn.Linear(dim_feats, 1000)
        self.c3 = nn.Sequential(
                nn.Linear(2000, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, num_classes)
                )


    def forward(self, x):
        x1 = x[:, :3, ...]
        x2 = x[:, 3:, ...]
        x1 = self.v1(x1)
        x2 = self.v2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.c3(out)
        return out

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    def forward(self, x, hidden):
        #print(x.size())
        x = x.view(-1, x.size(1))
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        outs = []
        hn = h0[0, :, :]
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self.fc(out)
        # out.size() --> 100, 10
        return out
    
class resGRU(nn.Module):


    def __init__(self,input_dim, hidden_dim, layer_dim, output_dim, bias=True,nfeaturenum = 1000,pretrained = True,backbone = 'resnext_ws50'):
        super(resGRU, self).__init__()
        # self.resmodel = l_resnext50()
        if backbone == 'Densenet':
            self.resmodel = pretrainedmodels.densenet121(num_classes=1000, pretrained='imagenet')
            self.resmodel.last_linear = nn.Sequential(nn.Linear(4096,1000))
            # print(self.resmodel)
            # self.resmodel.features.denseblock4.denselayer16.register_forward_hook(self.get_features_hook())
        elif backbone =='resnext_ws50':
            self.resmodel = l_resnext50()
            if pretrained:
                self.resmodel = init_with_pretrained(self.resmodel, 'resnext_ws50')

        # self.GRUModel = GRUModel(input_dim, hidden_dim, layer_dim, output_dim, bias=True)
        self.GRUModel = torch.nn.GRU(1000,1024,2,batch_first=True,dropout=0.25)
        self.fc = nn.Sequential(
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        if not hasattr(self, '_flattened'):
            self.GRUModel.flatten_parameters()
            setattr(self, '_flattened', True)
        x = x.permute(1,0,2,3,4)
        features = []

        for i in range(x.size(0)):
            features.append(self.resmodel(x[i,:,:,:,:]))

        features = torch.cat(features,dim = 0)
        # print(features.size())
        # features.permute(1,0,2)
        res,h = self.GRUModel(features.unsqueeze(0))

        return self.fc(res[:,-1,:])

class resLstm(nn.Module):
    def __init__(self,input_dim, hidden_dim, layer_dim, output_dim, bias=True,nfeaturenum = 1000,pretrained = True):
        super(resLstm, self).__init__()
        self.resmodel = models.resnet50(num_classes=nfeaturenum)
        if pretrained:
            model = init_with_pretrained(self.resmodel, 'resnet50')
        # self.GRUModel = GRUModel(input_dim, hidden_dim, layer_dim, output_dim, bias=True)
        self.lstm = torch.nn.LSTM(input_dim,hidden_dim,layer_dim,bias = True,batch_first=True,dropout=0.25,bidirectional=True)

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim*2*50, output_dim),
            nn.Sigmoid()
        )
    def forward(self,x):
        if not hasattr(self, '_flattened'):
            self.lstm.flatten_parameters()
            setattr(self, '_flattened', True)
        x = x.permute(1,0,2,3,4)
        features = []
        for i in range(x.size(0)):
            features.append(self.resmodel(x[i,:,:,:,:]))

        features = torch.cat(features,dim = 0)
        # print(features.size())
        # features.permute(1,0,2)
        output, (hn, cn) = self.lstm(features.unsqueeze(0))

        #print(output.size())
        b,s,f = output.size()

        return self.fc2(output.view(b,-1))


def get_model(model_name='resGRU', nclass=1000, pretrained=True):
    if model_name == 'DenseGRU':
        model = resGRU(1000,1024,3,1,backbone='Densenet')
    if model_name == 'resGRU':
        model = resGRU(1000,1024,3,1,backbone='resnext_ws50')
    if model_name == 'resnet18':
        model = models.resnet18(num_classes=nclass)
        if pretrained:
            model = init_with_pretrained(model, model_name)

    elif model_name == 'resnet34':
        model = models.resnet34(num_classes=nclass)
        if pretrained:
            model = init_with_pretrained(model, model_name)
    elif model_name == 'resnet50':
        model = models.resnet34(num_classes=nclass)
        if pretrained:
            model = init_with_pretrained(model, model_name)
    elif model_name == 'vgg16':
        #model = models.vgg16_bn(num_classes=nclass)
        model = vgg16_bn(num_classes=nclass)
        if pretrained:
            model = init_with_pretrained(model, model_name)
    elif model_name == 'googlenet':
        model = models.GoogLeNet(num_classes=nclass)
    elif model_name == 'shuffle_net':
        model = models.shufflenet_v2_x1_0(num_classes=nclass)
    elif model_name == 'mobile_net':
        model = models.mobilenet_v2(num_classes=nclass)
    elif model_name == 'se18':
        model = resnet18(num_classes=nclass)
    elif model_name == 'se50':
        model = resnet50(num_classes=nclass)
    elif model_name == 'vgg16x2':
        model = Vgg16X2(num_classes=nclass)
        if pretrained:
            pre_dict = torch.load(model_weight_dir['vgg16'])
            print("pre train:", list(pre_dict.keys())[:10])
            model_dict = model.state_dict()
            for k in model_dict:
                if 'classifier' in k or 'c3' in k or 'num_batches_tracked' in k:
                    continue
                # print(k, k[3:])
                model_dict[k] = pre_dict[k[3:]]
            model.load_state_dict(model_dict)
            print("Loaded pretrained {} model from {}.".format(
                model_name, model_weight_dir['vgg16']))
    elif model_name == 'res18x2WithOutDecoder':
        model = {}
        model['share'] = Res18X2WithoutDecoder()
        for i in range(nclass):
            model[str(i)] =AttributeDecoder()
        if pretrained:
            pre_dict = torch.load(model_weight_dir['resnet18'])
            #print(pre_dict)

            model_dict = model['share'].state_dict()
            #print(model_dict)
            for k in model_dict:
                if 'module.'+k in pre_dict.keys() and 'bn' not in k:
                    model_dict[k] = pre_dict['module.'+k]
                else:
                    print('not found')
            model['share'].load_state_dict(model_dict)

    elif model_name == 'res18x2':
        model = Res18X2(num_classes=nclass)
        if pretrained:
            pre_dict = torch.load(model_weight_dir['resnet18'])
            model.load_state_dict(pre_dict)
            # pre_dict = torch.load(model_weight_dir['resnet18'])
            #
            # print("pre train:", list(pre_dict.keys())[:10])
            # model_dict = model.state_dict()
            # for k in model_dict:
            #     if 'classifier' in k or 'c3' in k or 'num_batches_tracked' in k:
            #         continue
            #     # print(k, k[3:])
            #     model_dict[k] = pre_dict[k[3:]]
            # model.load_state_dict(model_dict)
            print("Loaded pretrained {} model from {}.".format(
                model_name, model_weight_dir['resnet18']))
    elif model_name == 'resnext50':
        # model = models.resnext50_32x4d(num_classes=nclass)
        # if pretrained:
        #     model = init_with_pretrained(model, model_name)
        # model_name = 'se_resnext50_32x4d'  # could be fbresnet152 or inceptionresnetv2
        # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        # dim_feats = model.last_linear.in_features  # =2048
        # model.avg_pool = nn.Sequential(nn.AvgPool2d(kernel_size=8, stride=8, padding=0),
        #                                nn.Conv2d(dim_feats, 2 * dim_feats, kernel_size=2, stride=1, padding=0),
        #                                nn.ReLU(),
        #                                nn.BatchNorm2d(2 * dim_feats))
        # model.last_linear = nn.Linear(2 * dim_feats, 2)
        model = Resnext50X2()
    return model


if __name__=='__main__':
    get_model(pretrained=True, model_name='resnext50')

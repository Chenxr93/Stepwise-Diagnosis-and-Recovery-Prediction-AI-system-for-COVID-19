import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.parameter import Parameter
import pretrainedmodels
from torch.autograd import Variable
from resnext_ws import l_resnext50
model_weight_dir = {
        'resnext_ws50':'X-50-GN-WS.pth.tar'
        }

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
    
class resGRU(nn.Module):
    def __init__(self,input_dim, hidden_dim, layer_dim, output_dim, bias=True,nfeaturenum = 1000,pretrained = True,backbone = 'resnext_ws50'):
        super(resGRU, self).__init__()

        if backbone == 'Densenet':
            self.resmodel = pretrainedmodels.densenet121(num_classes=1000, pretrained='imagenet')
            self.resmodel.last_linear = nn.Sequential(nn.Linear(4096,1000))
        elif backbone =='resnext_ws50':
            self.resmodel = l_resnext50()
            if pretrained:
                self.resmodel = init_with_pretrained(self.resmodel, 'resnext_ws50')
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

def get_model(model_name='resGRU', nclass=1000, pretrained=True):
    if model_name == 'DenseGRU':
        model = resGRU(1000,1024,3,1,backbone='Densenet')
    if model_name == 'resGRU':
        model = resGRU(1000,1024,3,1,backbone='resnext_ws50')
    return model


if __name__=='__main__':
    get_model(pretrained=True, model_name='resnext50')

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet101
from .densenet import densenet121
from .aspp import ASPP
from .decoder import Decoder
import pdb


class DeepLab(nn.Module):
    def __init__(self,):
        super(DeepLab, self).__init__()
        self.extractor = densenet121(pretrained=True)
                         #resnet101(pretrained=True,
                         #          replace_stride_with_dilation=[False, True, True])
        self.aspp = ASPP()
        self.decoder = Decoder(num_class=2)

    def forward(self, x):
        x, low_features, media_features, high_features = self.extractor(x)
        #pdb.set_trace()
        seg = self.aspp(media_features, high_features)
        seg = self.decoder(low_features, seg)
        #return x, seg
        return seg

if __name__ == '__main__':
    in_tensor = torch.zeros([2, 3, 512, 512])
    model = DeepLab()
    x, seg = model(in_tensor)
    print(in_tensor.shape, x.shape)

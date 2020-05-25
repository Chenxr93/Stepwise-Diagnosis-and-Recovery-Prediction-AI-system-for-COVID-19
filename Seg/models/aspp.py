import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class ASPPConv(nn.Sequential):
    def __init__(self, in_c, out_c, dilation):
        modules = [
            nn.Conv2d(in_c, out_c, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
            ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_c, out_c):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU() )

    def forward(self, x):
        size = x.size()[2:]
        for mod in self:
            x = mod(x)
            #print(x.shape)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_c=256, out_c=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()))

        r1, r2, r3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_c, out_c, r1))
        modules.append(ASPPConv(in_c, out_c, r2))
        modules.append(ASPPConv(in_c, out_c, r3))
        #modules.append(ASPPPooling(in_c, out_c))
        self.convs = nn.ModuleList(modules)

        self.highconv = nn.Sequential(
                nn.Conv2d(in_c * 4, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
                )
        self.out = nn.Sequential(
            nn.Conv2d(5 * out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout2d())

    def forward(self, x, high):
        s = x.size()[-1]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        high = F.interpolate(high, size=(s, s), mode='bilinear', align_corners=False)
        res.append(self.highconv(high))
        res = torch.cat(res, dim=1)
        res = self.out(res)
        return F.interpolate(res, size=(4*s, 4*s), mode='bilinear', align_corners=False)

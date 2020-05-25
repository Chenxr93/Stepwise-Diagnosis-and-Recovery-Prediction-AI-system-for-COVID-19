import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, in_c=64, out_c=48, num_class=2):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

        self.conv2 = nn.Sequential(
            nn.Conv2d(304, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            )
        self. conv3 = nn.Sequential(
            nn.Conv2d(256, num_class, 1),
            nn.ReLU()
            )

    def forward(self, low_features, x):

        s = low_features.size()[-1]
        low_features = self.conv1(low_features)
        low_features = self.bn1(low_features)
        low_features = self.relu(low_features)
        #print(x.shape, low_features.shape)

        x = F.interpolate(x, size=(s, s), mode='bilinear', align_corners=True)
        x = torch.cat([x, low_features], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.interpolate(x, size=(s * 4, s * 4), mode='bilinear', align_corners=True)

        return x

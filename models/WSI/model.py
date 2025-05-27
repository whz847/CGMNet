import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from models.WSI.ViT_2D import VisionTransformer, PatchEmbed_2d


class CroMAM(nn.Module):
    def __init__(
            self,
            backbone='resnet18',
            pretrained=True,
            output_features=False,
            device=torch.device("cuda:1"),
            args=None
    ):
        super(CroMAM, self).__init__()

        self.output_features = output_features
        self.device = device

        # initialize backbone model
        if isinstance(backbone, str):
            if pretrained:
                # self.backbone1 = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
                # self.backbone2 = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
                self.backbone1 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                # self.backbone1 = torchvision.models.swin_t(weights=None)
                # self.backbone2 = torchvision.models.swin_t(weights=None)
                self.backbone1 = torchvision.models.resnet18(weights=None)
            # backbone_nlayers = 0
            # model_dim = {'swin_transformer': {0: 768}}
            # feature_dim = model_dim[backbone][backbone_nlayers]

            self.layergetter1 = torchvision.models._utils.IntermediateLayerGetter(self.backbone1,
                                                                                  {'layer1': 'feat1', 'layer2': 'feat2',
                                                                                   'layer3': 'feat3',
                                                                                   'layer4': 'feat4'})
            self.conv64_512 = nn.Conv2d(64, 512, kernel_size=1)
            self.conv128_512 = nn.Conv2d(128, 512, kernel_size=1)
            self.conv256_512 = nn.Conv2d(256, 512, kernel_size=1)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, 128)

            # 获取 ResNet-18 最后一层的输入特征数
            num_features = self.backbone1.fc.in_features  # 512
            feature_dim = num_features
        else:
            pass

        # self.backbone1.head = nn.Identity()
        # self.backbone2.head = nn.Identity()
        self.backbone1.fc = nn.Identity()

        self.vit_2d = VisionTransformer(img_size=8,
                                        in_c=512,
                                        patch_size=8,
                                        embed_dim=2048,
                                        depth=4,
                                        num_heads=8,
                                        num_classes=4,
                                        embed_layer=PatchEmbed_2d)

    def forward(self, x1=None):
        # 获取各个维度的大小
        b, c, h, w = x1.size()
        # inter_feats1 = self.layergetter1(x1)
        # feat1 = self.gap(F.relu(self.conv64_512(inter_feats1["feat1"])))
        # feat2 = self.gap(F.relu(self.conv128_512(inter_feats1["feat2"])))
        # feat3 = self.gap(F.relu(self.conv256_512(inter_feats1["feat3"])))
        # feat4 = self.gap(inter_feats1["feat4"])
        #
        # feat1_aggregated = feat1.view(-1, b, feat1.shape[1], feat1.shape[2], feat1.shape[3]).mean(dim=1)
        # feat2_aggregated = feat2.view(-1, b, feat2.shape[1], feat2.shape[2], feat2.shape[3]).mean(dim=1)
        # feat3_aggregated = feat3.view(-1, b, feat3.shape[1], feat3.shape[2], feat3.shape[3]).mean(dim=1)
        # feat4_aggregated = feat4.view(-1, b, feat4.shape[1], feat4.shape[2], feat4.shape[3]).mean(dim=1)
        # feature = feat1_aggregated + feat2_aggregated + feat3_aggregated + feat4_aggregated
        # feature = feature.squeeze(2)
        # feature = feature.squeeze(2)
        x = self.backbone1(x1)
        x = x.view(-1, b, x.shape[1]).mean(dim=1)

        x = self.fc(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    with torch.no_grad():
        x1 = torch.randn((64, 3, 256, 256), device=device)
        # Read parameter configuration
        model = CroMAM(
            backbone='resnet18',
            pretrained=True,
            device=device,
        )
        model.cuda()
        output = model(x1)
        print(output.shape)

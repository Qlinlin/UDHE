
import torch
import torch.nn as nn
from einops import rearrange


## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)




def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return  torch.cat([x_HL, x_LH, x_HH], dim=1)



class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out




def rgb_to_gray(rgb):
    return 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]

# HF-ERB
class EfficientResidualBlock(nn.Module):
    def __init__(self, channels, groups=2):
        super().__init__()
        self.main_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups, padding_mode='reflect'),
            nn.InstanceNorm2d(channels)
        )
        self.dwt = DWT()
        self.attention = Attention(channels,8,False)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 1x1卷积用于将3倍通道数降回原始通道数
        self.channel_adjust = nn.Conv2d(
            channels * 3,  # 输入通道数：原始通道数×3（因为拼接了3个高频分量）
            channels,  # 输出通道数：与输入保持一致
            kernel_size=1,  # 1x1卷积不改变空间尺寸
            stride=1,
            padding=0
        )
    def forward(self, x):
        res = self.main_conv(x)
        high_freq = self.dwt(x)
        res = res + 1.0 * self.channel_adjust(self.upsample(high_freq))
        res = self.attention(res)
        return self.relu(x + res)




#  简化分支
class LightweightBranch(nn.Module):
    def __init__(self, in_channels, base_channels=12):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1, padding_mode='reflect')
        self.res_blocks = nn.Sequential(
            EfficientResidualBlock(base_channels),
            EfficientResidualBlock(base_channels)
        )
        self.low_scale = nn.Conv2d(base_channels, base_channels, 1)
        self.high_scale = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            EfficientResidualBlock(base_channels * 2)
        )

        self.cross_scale_attn = CrossScaleAttention(base_channels,base_channels*2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.init_conv(x)
        res_feat = self.res_blocks(x)
        low_feat = self.low_scale(res_feat)
        high_feat = self.high_scale(res_feat)

        high_feat_up = self.upsample(high_feat)

        low_feat = self.cross_scale_attn(low_feat, high_feat_up)

        return {
            'low': low_feat,
            'high': high_feat,
            'high_up': high_feat_up
        }


# CSA
class CrossScaleAttention(nn.Module):
    def __init__(self, low_channels, high_channels):
        super().__init__()
        self.high2low = nn.Conv2d(high_channels, low_channels, 1, bias=False)  # 将high_feat通道降到low_feat水平
        self.map_feat = nn.Conv2d(low_channels * 2, low_channels, 1, bias=False)  # 拼接后降维
        self.attn = nn.Sequential(
            nn.Conv2d(low_channels, low_channels, 3, padding=1, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, low_feat, high_feat_up):
        assert low_feat.shape[2:] == high_feat_up.shape[2:], \
            f"高低尺度特征尺寸不匹配：low_feat {low_feat.shape[2:]}, high_feat_up {high_feat_up.shape[2:]}"

        high_feat_up = self.high2low(high_feat_up)  # 通道数从high_channels → low_channels

        concat_feat = torch.cat([low_feat, high_feat_up], dim=1)
        guide_weight = self.attn(self.map_feat(concat_feat))

        # 动态融合：low_feat和调整后的high_feat_up按权重融合
        return low_feat * guide_weight + high_feat_up * (1 - guide_weight)
# fusion
class SimpleFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.color2gray_attn = nn.Conv2d(channels, channels, 1)
        self.gray2color_attn = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, color_feat, gray_feat):
        color_attn = color_feat * torch.sigmoid(self.gray2color_attn(gray_feat))
        gray_attn = gray_feat * torch.sigmoid(self.color2gray_attn(color_feat))
        gate = self.gate(torch.cat([color_attn, gray_attn], dim=1))
        return gate * color_attn + (1 - gate) * gray_attn


# 5. full model
class LightweightRestorationNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.base_channels = base_channels  # 保存基础通道数用于计算
        self.color_branch = LightweightBranch(in_channels=3, base_channels=base_channels)
        self.gray_branch = LightweightBranch(in_channels=1, base_channels=base_channels)

        self.fuse_low = SimpleFusion(base_channels)
        self.fuse_high = SimpleFusion(base_channels * 2)

        # 聚合器输入通道数 = 低尺度通道数 + 高尺度通道数
        self.aggregator = nn.Sequential(
            nn.Conv2d(base_channels + base_channels * 2, base_channels * 2, 3, padding=1, padding_mode='reflect'),
            EfficientResidualBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, 3, 3, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, rgb):
        gray = rgb_to_gray(rgb)

        color_feats = self.color_branch(rgb)
        gray_feats = self.gray_branch(gray)

        fused_low = self.fuse_low(color_feats['low'], gray_feats['low'])  # 通道数: base_channels
        fused_high = self.fuse_high(color_feats['high'], gray_feats['high'])  # 通道数: base_channels*2
        fused_high_up = self.color_branch.upsample(fused_high)  # 通道数保持base_channels*2

        # 拼接后的总通道数: base_channels + base_channels*2
        aggregated = torch.cat([fused_low, fused_high_up], dim=1)
        restored = self.aggregator(aggregated)
        return (restored + 1) / 2




# 测试
if __name__ == "__main__":

    import torchinfo
    torchinfo.summary(LightweightRestorationNet(base_channels=32), input_size=(4,3, 256, 256),col_names=['input_size','output_size','num_params','trainable'],
                      col_width=20,
                      row_settings=['var_names'])

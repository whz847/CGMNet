import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MLCA3D(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(MLCA3D, self).__init__()

        # ECA 计算方法
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight = local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool3d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)

        b, c, d, m, n = x.shape  # 注意这里的维度顺序是(batch, channel, depth, height, width)
        _, _, d_local, m_local, n_local = local_arv.shape

        # (b,c,d,local_size,local_size) -> (b,c,d*local_size*local_size)-> (b,d*local_size*local_size,c)-> (b,1,d*local_size*local_size*c)
        temp_local = local_arv.view(b, c, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)

        # (b,c,d,local_size,local_size) <- (b,c,d*local_size*local_size)<-(b,d*local_size*local_size,c) <- (b,1,d*local_size*local_size*c)
        y_local_transpose = y_local.reshape(b, d_local * m_local * n_local, c).transpose(-1, -2).view(b, c, d_local,
                                                                                                      m_local, n_local)
        y_global_transpose = y_global.view(b, -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 增加一个维度以适应3D张量

        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool3d(y_global_transpose.sigmoid(), [d_local, m_local, n_local])
        att_all = F.adaptive_avg_pool3d(att_global * (1 - self.local_weight) + (att_local * self.local_weight),
                                        [d, m, n])

        x = x * att_all
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.rand(1, 128, 16, 16, 16).to(device)  # 修改了输入张量形状以匹配3D数据
    attention_module = MLCA3D(128).to(device)
    output_tensor = attention_module(input_tensor)
    # 打印输入和输出的形状
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
import math
import torch
import torch.nn as nn

# SiLU激活函数: SiLU(x) = x * sigmoid(x)
class SiLU(nn.Module):
    """
    SiLU激活函数,也称为Swish激活函数.
    SiLU(x) = x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)
    
# 32位精度的GroupNorm
class GroupNorm32(nn.GroupNorm):
    """
    一个32位精度的GroupNorm层,这对于在混合精度训练中防止数值不稳定性很有用,因为GroupNorm在低精度下可能会计算不稳定.
    """
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
# 创建一个1D, 2D或3D卷积层
def conv_nd(dims, *args, **kwargs):
    """
    根据指定的维度创建一个N维卷积层.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"不支持的维度: {dims}")

# 线性层
def linear(*args, **kwargs):
    """
    创建一个线性层.
    """
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    """
    根据指定的维度创建一个N维平均池化层.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"不支持的维度: {dims}")

def update_ema(target_model, source_model, beta_rate=0.99):
    """
    使用指数移动平均(EMA)更新模型参数.
    target_model: 目标EMA模型
    source_model: 源模型
    beta_rate: EMA衰减率
    公式: ema_param = beta_rate * ema_param + (1 - beta_rate) * param
    作用: 扩散模型通常使用 EMA 版本的权重来生成图像，这样生成的图像质量更稳定、更好。这部分代码会在训练循环中被频繁调用。
    """
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.mul_(beta_rate).add_(source_param.data, alpha=1 - beta_rate) 

def zero_module(module):
    """
    将一个模块的所有参数（权重和偏置）归零。在残差连接的末端，通常会把最后一层卷积初始化为 0。这样模型初始状态下就真的是“恒等映射”（Identity），对深层网络的训练非常友好。
    """
    for param in module.parameters():
        param.detach().zero_()
    return module

def scale_module(module, scale):
    """
    将一个模块的所有参数（权重和偏置）按指定比例缩放。
    """
    for param in module.parameters():
        param.data.mul_(scale)
    return module

def mean_flat(tensor):
    """
    计算张量在除批次维度外的所有维度的均值.
    """
    return tensor.mean(dim=tuple(range(1, len(tensor.shape))))

def normalization(channels):
    """
    创建一个32组的GroupNorm归一化层.
    """
    return GroupNorm32(num_groups=32, num_channels=channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    为时间步生成正弦余弦嵌入.
    timesteps: 张量,形状为 [N], 包含时间步索引
    dim: 嵌入维度
    max_period: 控制频率范围的最大周期
    返回: 形状为 [N, dim] 的嵌入张量
    例子:
    >>> t = torch.tensor([1, 2, 3])
    >>> emb = timestep_embedding(t, dim=6)
    >>> print(emb)
    tensor([[ 0.9999,  0.9950,  0.9801,  0.0099,  0.0998,  0.1987],
            [ 0.9998,  0.9801,  0.9211,  0.0175,  0.1987,  0.3894],
            [ 0.9995,  0.9553,  0.8415,  0.0299,  0.2955,  0.5403]])
    解释: 每个时间步都被映射到一个由正弦和余弦函数组成的高维空间中,有助于模型捕捉时间步之间的关系.
    公式参考: https://arxiv.org/abs/1706.03762 (Transformer论文中的位置编码)
    公式: embedding[i, 2j] = cos(timesteps[i] / (max_period^(2j/dim)))
          embedding[i, 2j+1] = sin(timesteps[i] / (max_period^(2j/dim)))
    其中 j 是维度索引.
    作用: 这种嵌入方式允许模型更好地理解时间步之间的相对关系,而不仅仅是绝对位置.
    """
    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    ) # 数学公式: exp(-log(max_period) * (i / half_dim)) for i in [0, half_dim-1]
    args = timesteps[:, None].float() * frequencies[None, :] # 广播
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:  # 如果维度是奇数,则在末尾添加一个零
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def checkpoint(func, inputs, params, flag):
    """
    检查点函数，用于在前向传播时节省内存，通过在反向传播时重新计算中间结果来实现。
    func: 需要检查点的函数
    inputs: 传递给函数的输入张量列表
    params: 传递给函数的参数张量列表
    flag: 布尔值，指示是否启用检查点功能
    返回: 函数的输出张量
    作用: 在训练大型神经网络时，内存可能成为瓶颈。通过使用检查点技术，可以在前向传播时节省内存，从而允许训练更大的模型或使用更大的批次大小。
    参考: https://pytorch.org/docs/stable/checkpoint.html
    """
    if flag:
        args = tuple(inputs) + tuple(params) # 将输入张量和参数张量连接成一个元组
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(torch.autograd.Function):
    """
    自定义的检查点函数，用于在前向传播时节省内存，通过在反向传播时重新计算中间结果来实现。
    继承自 torch.autograd.Function，需要实现 forward 和 backward 方法。
    作用: 在训练大型神经网络时，内存可能成为瓶颈。通过使用检查点技术，可以在前向传播时节省内存，从而允许训练更大的模型或使用更大的批次大小。
    参考: https://pytorch.org/docs/stable/checkpoint.html
    """
    @staticmethod
    def forward(ctx, run_function, length, *args):
        """
        前向传播函数，执行计算并保存必要的上下文信息以供反向传播使用。
        """
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = run_function(*ctx.input_tensors)
        return output_tensors
    @staticmethod
    def backward(ctx, *output_grads):
        """
        反向传播函数，计算梯度。
        """
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            outputs=output_tensors,
            inputs=ctx.input_tensors + ctx.input_params,
            grad_outputs=output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads






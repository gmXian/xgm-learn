import numpy as np
import torch as th

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two normal distributions.
    
    Args:
        mean1: Mean of the first normal distribution.
        logvar1: Log variance of the first normal distribution.
        mean2: Mean of the second normal distribution.
        logvar2: Log variance of the second normal distribution.
        
    Returns:
        KL divergence between the two distributions.
    """
    # 捕获输入中的某一个 Tensor 实例。这样后续调用 .to(tensor) 时，可以将新创建的标量 Tensor 自动移动到和输入数据相同的设备（CPU/GPU）和数据类型上。
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj 
            break
    assert tensor is not None, "At least one input must be a tensor."

    logvar1, logvar2 = [
        x if isinstance(x, tensor.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]
    # 下面的公式是: KL(N(mean1, var1) || N(mean2, var2)) = 0.5 * ( -1 + log(var2) - log(var1) + var1/var2 + ((mean1-mean2)^2)/var2 )
    return 0.5 * (-1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) +((mean1-mean2)**2)*th.exp(-logvar2))

def approx_standard_normal_cdf(x):
    """
    Approximate the CDF of the standard normal distribution. 近似标准正态分布的累积分布函数
    公式是: 0.5 * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def discretized_gaussian_log_likelihood(x,*, means, log_scales):
    """
    Compute the log-likelihood of `x` under a discretized Gaussian distribution.
    
    Args:
        x: Input data.
        means: Means of the Gaussian distribution.
        log_scales: Log scales of the Gaussian distribution.
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
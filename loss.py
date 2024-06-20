import torch.nn.functional as F
import torch
import numpy as np
from scipy.stats import norm
import math
import sys


# mse loss
def mse_loss(predict, target):
    loss = torch.nn.MSELoss()
    return loss(predict, target)


def kl_loss(z_mean, z_stddev):
    return torch.mean(-0.5 * torch.sum(1 + 2*torch.log(z_stddev) - z_mean ** 2 - z_stddev ** 2, dim=1), dim=0)


def entropy_loss(p, eps=1e-8):
    log_probs = torch.log(p + eps)
    tmp = p * log_probs
    entropy = - torch.sum(tmp, dim=1)
    # print(entropy.size())
    return torch.sum(entropy, dim=1)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def log_truncated_normal(x, mu, sigma, eps=1e-8):
    # sigma = torch.minimum(sigma, torch.tensor(1e3))

    var = (sigma ** 2)
    log_scale = torch.log(sigma + eps)
    tmp = torch.tensor(math.pi, device='cuda')
    return -((x - mu) ** 2) / (2 * var) - log_scale - torch.log(torch.sqrt(2 * tmp))


def normal_loss(x, mu, sigma):
    ll = log_truncated_normal(x, mu, sigma)
    result = - torch.mean(ll)
    # result = _nan2inf(result)
    return result


def tobit_loss(x, mu, sigma, eps=1e-8):
    ll1 = log_truncated_normal(x, mu, sigma)
    cdf = np.float32(norm.cdf(-mu.cpu().detach().numpy()/sigma.cpu().detach().numpy()))
    ll2 = torch.log(torch.tensor(cdf, device=mu.device)+eps)
    tmp = torch.where(x > 0, ll1, ll2)
    result = - torch.mean(tmp)
    return result


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    # x = x.float()

    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    softplus_pi = F.softplus(-pi)

    log_theta_eps = torch.log(theta + eps)

    log_theta_mu_eps = torch.log(theta + mu + eps)

    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (-softplus_pi + pi_theta_log
                     + x * (torch.log(mu + eps) - log_theta_mu_eps)
                     + torch.lgamma(x + theta)
                     - torch.lgamma(theta)
                     - torch.lgamma(x + 1))

    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    result = - torch.sum(res, dim=1)
    result = _nan2inf(result)

    return result


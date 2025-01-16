import torch
import math
import random
import os
import numpy as np
from math import exp, sqrt
from scipy.special import erf

# DP函数，输入某个weight/bias:parameter
# 输出扰乱之后（应用DP）之后的参数
def PM(net, privacy_budget):
    with torch.no_grad():
        flag = 0
        for name, parameters in net.named_parameters():
            if "bias" in name or "norm" in name:
                continue
            else:
                original_shape = parameters.shape
                level_parameters = parameters.flatten()
                radius = (level_parameters.max() - level_parameters.min()) / 2
                random_nums = np.random.uniform(0, 1, len(level_parameters))
                K = (math.exp(privacy_budget / 2) + 1) / (math.exp(privacy_budget / 2) - 1)
                left = ((K + 1) * level_parameters) / 2 - (radius * (K - 1)) / 2
                right = left + radius * (K - 1)
                compare = random_nums < (1 / (math.exp(privacy_budget / 2) + 1))
                for i in range(len(level_parameters)):
                    if level_parameters[i] == 0:
                        level_parameters[i] = 0
                    else:
                        if compare[i]:
                            numbers = list()
                            numbers.append(random.uniform(-radius * K, left[i]))
                            numbers.append(random.uniform(right[i], radius * K))
                            level_parameters[i] = numbers[random.randint(0, 1)]
                        else:
                            level_parameters[i] = random.uniform(left[i], right[i])
                level_parameters = level_parameters.view(original_shape)
                parameters.copy_(level_parameters)
            flag = flag + 1
            if flag == 2:
                break
    return net


def PM_MNIST(net, privacy_budget, K):
    flag = 0
    all_para = []
    for name, parameters in net.named_parameters():
        if parameters.requires_grad and flag % 2 == 0 and flag < 3:
            print("++++++++", name)
            # level_parameters = parameters.view(1, -1)
            level_parameters = parameters.cpu().detach().numpy().flatten()
            print(level_parameters)
            radius = level_parameters.max()
            random_nums = np.random.uniform(0, 1, len(level_parameters))
            left = ((K + 1) * level_parameters) / 2 - (radius * (K - 1)) / 2
            right = left + radius * (K - 1)
            compare = random_nums < (1 / (math.exp(privacy_budget / 2) + 1))
            for i in range(len(level_parameters)):
                if compare[i]:
                    numbers = list()
                    numbers.append(random.uniform(-radius * K, left[i]))
                    numbers.append(random.uniform(right[i], radius * K))
                    level_parameters[i] = numbers[random.randint(0, 1)]
                else:
                    level_parameters[i] = random.uniform(left[i], right[i])
            print(level_parameters)
            level_parameters[level_parameters > radius] = radius
            level_parameters[level_parameters < -radius] = -radius
            print(level_parameters)
            all_para.append(level_parameters)
        flag = flag + 1
    return all_para

def OPM_MNIST(net, privacy_budget, K):
    flag = 0
    all_para = []
    for name, parameters in net.named_parameters():
        if parameters.requires_grad and flag % 2 == 0 and flag < 3:
            print("++++++++", name)
            # level_parameters = parameters.view(1, -1)
            level_parameters = parameters.cpu().detach().numpy().flatten()
            print(level_parameters)
            radius = level_parameters.max()
            random_nums = np.random.uniform(0, 1, len(level_parameters))
            left = ((K + 1) * level_parameters) / 2 - (radius * (K - 1)) / 2
            right = left + radius * (K - 1)
            compare = random_nums < (1 / (math.exp(privacy_budget / 2) + 1))
            for i in range(len(level_parameters)):
                if compare[i]:
                    numbers = list()
                    numbers.append(random.uniform(-radius * K, left[i]))
                    numbers.append(random.uniform(right[i], radius * K))
                    level_parameters[i] = numbers[random.randint(0, 1)]
                else:
                    level_parameters[i] = random.uniform(left[i], right[i])
            print(level_parameters)
            level_parameters[level_parameters > radius] = radius
            level_parameters[level_parameters < -radius] = -radius
            print(level_parameters)
            all_para.append(level_parameters)
        flag = flag + 1
    return all_para

def DP_laplace(net, privacy_budget):
    flag = 0
    # sensitivity = 1
    all_para = []
    for name, parameters in net.named_parameters():
        if parameters.requires_grad and flag % 2 == 0 and flag < 3:
            level_parameters = parameters.cpu().detach().numpy().flatten()
            level_parameters[level_parameters > 0.5] = 0.5
            level_parameters[level_parameters < -0.5] = -0.5
            # np.random.laplace可以获得拉普拉斯分布的随机值，参数主要如下：
            # loc：就是上面的μ，控制偏移。scale： 就是上面的λ控制缩放。 size：  是产生数据的个数
            # print("max--", max(level_parameters), "min--", min(level_parameters))
            sensitivity = 1
            scale_noise = sensitivity / privacy_budget
            Laplace_noise = np.random.laplace(0, scale_noise, len(level_parameters))
            level_parameters = level_parameters + Laplace_noise
            all_para.append(level_parameters)
        flag = flag + 1
    return all_para

def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol=1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]
    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)
    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))

    def caseA(epsilon, s):
        return Phi(sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

    def caseB(epsilon, s):
        return Phi(-sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while (not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while (not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s: caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s: caseA(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)

        else:
            predicate_stop_DT = lambda s: caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s: caseB(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

        predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    sigma = alpha * GS / sqrt(2.0 * epsilon)

    return sigma

def DP_Gaussian(net, privacy_budget, delta=0.01, GS=1):
    flag = 0
    # sensitivity = 1
    all_para = []
    for name, parameters in net.named_parameters():
        if parameters.requires_grad and flag % 2 == 0 and flag < 3:
            level_parameters = parameters.cpu().detach().numpy().flatten()
            level_parameters[level_parameters > 0.5] = 0.5
            level_parameters[level_parameters < -0.5] = -0.5
            # np.random.laplace可以获得拉普拉斯分布的随机值，参数主要如下：
            # loc：就是上面的μ，控制偏移。scale： 就是上面的λ控制缩放。 size：  是产生数据的个数
            # print("max--", max(level_parameters), "min--", min(level_parameters))
            sigma = calibrateAnalyticGaussianMechanism(privacy_budget, delta, GS)
            Gaussian_noise = np.random.normal(0, sigma * sigma, len(level_parameters))
            level_parameters = level_parameters + Gaussian_noise
            all_para.append(level_parameters)
        flag = flag + 1
    return all_para
import torch
import matplotlib.pyplot as plt
import random
import numpy as np

#problem 1
# 1.p86 sk=0时
# Pk|k−1 6= E[(xk − ˆxkf|k−1)(xk − ˆxkfk|k−1)0]
# 估计值（解析解）等于真实值（随机数生成后迭代计算pk）
# 其中sk(七个变量） -> F/H/G/V(Q.R)/x0~N(jun,p0)


import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.distributions.normal as normal


a,b,c=torch.rand(3,20,20)

print(a)
print(b)
print(c)


a=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
b=torch.tensor([1.0,2.0,3.0])
# print(torch.inverse(a))
print(torch.mean(a,1))
print(a+3)

# a=torch.tensor([1.0,2.0,3.0])
# b=torch.tensor([5.0,6.0,7.0])
# print(a*b)

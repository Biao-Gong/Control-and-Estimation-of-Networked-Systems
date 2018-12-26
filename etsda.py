import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.distributions.normal as normal


# a,b,c=torch.rand(3,20,20)

# print(a)
# print(b)
# print(c)


# a=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
# b=torch.tensor([[1.0,2.0],[3.0,5.0]])
# c=torch.tensor([[2.0]])
# print(c+a)
# normal.Normal(torch.zeros(2,1), torch.tensor([0.1]))
# print(normal.Normal(torch.zeros(2,1), 0.1).sample())

a=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
a[:,0]=0.0
print(a)

# d=torch.tensor([[1.0],[2.0],[3.0],[5.0]])
# e=torch.tensor([[1.0,0.0,0.1,0.0],[0.0,1.0,0.0,0.1],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
# print(d+e)

# print(torch.inverse(a))

# print(d.mm(e.reshape(1,-1)))

# a=torch.tensor([1.0,2.0,3.0])
# b=torch.tensor([5.0,6.0,7.0])
# print(a*b)


import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.distributions.normal as normal
import math

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

S0=True         # WHEN S0=0 IS TRUE
n=4             # length of X0
Hn=2
expnumb=10000    # expr numb
group=500       # group number

########################################
X0ba=torch.tensor([[0.0],[0.0],[2.0],[2.0]])
P=1.5
P0=torch.eye(n)*P
Q=0.1
Q0=torch.eye(n)*Q
R=0.12
R0=torch.eye(Hn)*R
Ts=0.01
F=torch.tensor([[1.0,0.0,Ts,0.0],
                [0.0,1.0,0.0,Ts],
                [0.0,0.0,1.0,0.0],
                [0.0,0.0,0.0,1.0]])
G=torch.tensor([[0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0],
                [0.0,0.0,Ts*Ts,0.0],
                [0.0,0.0,0.0,Ts*Ts]])
H=torch.tensor([[1.0,0.0,0.0,0.0],
                [0.0,1.0,0.0,0.0]])
########################################
nX0=normal.Normal(X0ba, math.sqrt(P))
if S0:
    nWk=normal.Normal(torch.zeros(n,1), math.sqrt(Q))
    nVk=normal.Normal(torch.zeros(Hn,1), math.sqrt(R))

Wk=nWk.sample()
Vk=nVk.sample()

# a=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
# b=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])

c=[[1,2,3],torch.tensor([1.0,2.0,3.0]),[4,5,6]]
L=torch.tensor([[1.0,0.0,0.0,0.0,0.0,0.0],
                [1.0,2.0,0.0,0.0,0.0,0.0],
                [1.0,2.0,3.0,0.0,0.0,0.0],
                [1.0,2.0,3.0,4.0,0.0,0.0],
                [1.0,2.0,3.0,4.0,2.0,0.0],
                [1.0,2.0,3.0,4.0,1.0,3.0]])
d=torch.cat((Wk.reshape(-1),Vk.reshape(-1))).reshape(-1,1)
print(L.mm(d))
print(L.mm(d)[:n,0].reshape(-1,1))
print(L.mm(d)[n:Hn+n,0].reshape(-1,1))
# print(torch.tensor(c))

# num_st=2
# num_st=1
# L=torch.tensor()
# temp=torch.tensor([Wk.t(),Vk.t()]*L.t()).t()

# d=torch.tensor([[1.0],[2.0],[3.0],[5.0]])
# e=torch.tensor([[1.0,0.0,0.1,0.0],[0.0,1.0,0.0,0.1],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
# print(d+e)

# print(torch.inverse(a))

# print(d.mm(e.reshape(1,-1)))

# a=torch.tensor([1.0,2.0,3.0])
# b=torch.tensor([5.0,6.0,7.0])
# print(a*b)


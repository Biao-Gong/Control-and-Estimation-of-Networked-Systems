import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.distributions.normal as normal
import math

if __name__ == '__main__':

    S0=True     # WHEN S0=0 IS TRUE
    n=4        # length of X0
    Hn=2
    expnumb=1000  # expr numb
    group=500    # group number

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
    else:
        pass
    ########################################
    #start
    X=torch.zeros(expnumb,group,n,1)
    Xyuce=torch.zeros(expnumb,group,n,1)
    Zyuce=torch.zeros(expnumb,group,Hn,1)
    Xkk=torch.zeros(expnumb,group,n,1)
    Pkk=torch.zeros(expnumb,group,n,n)

    Xkk[:,0]=nX0.sample()
    Pkk[:,0]=P0
    Xyuce[:,0]=X0ba
    Zyuce[:,0]=H.mm(X0ba)+nVk.sample()

    result=torch.zeros(expnumb,n,n)
    resultPkkjy=torch.zeros(expnumb,n,n)
    montecarlo=torch.zeros(expnumb)
    ########################################
    for i in range(expnumb):
        for j in range(group-1):

            Wk=nWk.sample()
            Vk=nVk.sample()

            # function
            Xyuce[i,j+1]=F.mm(Xyuce[i,j])+G.mm(Wk)     # X1 by X0
            Zyuce[i,j+1]=H.mm(Xyuce[i,j+1])+Vk         # Z1 by X0            

            # kalman
            Kk=Pkk[i,j].mm(H.t()).mm(torch.inverse(H.mm(Pkk[i,j]).mm(H.t())+R0))
            Xkk[i,j+1]=Xkk[i,j]+Kk.mm(Zyuce[i,j]-H.mm(Xkk[i,j]))
            Pkk[i,j+1]=Pkk[i,j]-Kk.mm(H).mm(Pkk[i,j])

            # time update
            Xkk[i,j+1]=F.mm(Xkk[i,j+1])                                 # size is n*1
            Pkk[i,j+1]=F.mm(Pkk[i,j+1]).mm(F.t())+G.mm(Q0).mm(G.t())    # size is n*n

        
        # DI I CI SHIYAN
        result[i]=(Xyuce[i,-1]-Xkk[i,-1]).mm((Xyuce[i,-1]-Xkk[i,-1]).t())
        resultPkkjy[i]=Pkk[i,-1]
        montecarlo[i]=torch.dist(torch.mean(result[:i+1],0),resultPkkjy[i],p=2)/torch.dist(resultPkkjy[i],torch.zeros(n,n),p=2)


    plt.subplot(211)
    plt.plot(Xyuce[i,::10,0,0].numpy(),Xyuce[i,::10,1,0].numpy(),'o-')
    plt.plot(Xkk[i,::10,0,0].numpy(),Xkk[i,::10,1,0].numpy(),'o-')

    plt.subplot(212)
    plt.plot(montecarlo.numpy()[::10],'o-')
    plt.show()





        

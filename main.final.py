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
    expnumb=1  # expr numb
    group=2000    # group number

    ########################################
    X0ba=torch.tensor([[0.0],[0.0],[1.0],[1.0]])
    P=0.5
    P0=torch.eye(n)*P
    Q=1.001
    Q0=torch.eye(n)*Q
    R=0.02
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
    Xkk_save=torch.zeros(expnumb,group,n,1)
    Pkk_save=torch.zeros(expnumb,group,n,n)

    result=torch.zeros(expnumb,n,n)
    resultPkkjy=torch.zeros(expnumb,n,n)
    montecarlo=torch.zeros(expnumb)
    ########################################
    for i in range(expnumb):

        Xkk=X0ba
        Pkk=P0
        X[i,0]=nX0.sample()             # X0 n*1
        # X[i,0]=math.sqrt(P)*torch.randn(n,1)
        Xyuce[i,0]=X0ba                 # X0
        
        for j in range(group-1):

            Wk=nWk.sample()
            Vk=nVk.sample()
            # Wk=math.sqrt(Q)*torch.randn(n,1)
            # Vk=math.sqrt(R)*torch.randn(Hn,1)

            # function
            X[i,j+1]=F.mm(X[i,j])+G.mm(Wk)             # X1 by X0 n*1
            Xyuce[i,j+1]=F.mm(Xyuce[i,j])+G.mm(Wk)     # X1 by X0
            Zyuce[i,j]=H.mm(Xyuce[i,j])+Vk             # Z0 by X0            

            # kalman
            # print(j)
            Kk=Pkk.mm(H.t()).mm(torch.inverse(H.mm(Pkk).mm(H.t())+R0))
            Xkk=Xkk+Kk.mm(Zyuce[i,j]-H.mm(Xkk))
            Pkk=Pkk-Kk.mm(H).mm(Pkk)

            # save
            Xkk_save[i,j]=Xkk
            Pkk_save[i,j]=Pkk

            # time update
            Xkk=F.mm(Xkk)                                 # size is n*1
            Pkk=F.mm(Pkk).mm(F.t())+G.mm(Q0).mm(G.t())    # size is n*n

            # if j==0:

            #     X[i,j]=F.mm(nX0.sample())+G.mm(Wk)             # X1 by X0 n*1
            #     Xyuce[i,j]=F.mm(X0ba)+G.mm(Wk)                 # X1 by X0
            #     Zyuce[i,j]=H.mm(Xyuce[i,j])+Vk                 # Z1 by X1

            #     # measurement update, X00 and P00 (need Z0 by X0)
            #     Kk=P0.mm(H.t()).mm(torch.inverse(H.mm(P0).mm(H.t())+R0))  # size is n*Hn
            #     Z0=H.mm(X0ba)+Vk                                          # size is Hn*1

            #     Xkk=X0ba+Kk.mm(Z0-H.mm(X0ba))                             # size is n*1
            #     Pkk=P0-Kk.mm(H).mm(P0)                                    # size is n*n

            # else:
                
            #     X[i,j]=F.mm(X[i,j-1])+G.mm(Wk)                       # X2 by X1 n*1
            #     Xyuce[i,j]=F.mm(Xyuce[i,j-1])+G.mm(Wk)               # X2 by X1 n*1
            #     Zyuce[i,j]=H.mm(Xyuce[i,j])+Vk                       # Z2 by X2 Hn*1

            #     # time update
            #     Xyuce_temp=F.mm(Xkk)                                 # size is n*1
            #     Pyuce_temp=F.mm(Pkk).mm(F.t())+G.mm(Q0).mm(G.t())    # size is n*n

            #     # measurement update X11 and P11 (need Z1)
            #     Kk=Pyuce_temp.mm(H.t()).mm(torch.inverse(H.mm(Pyuce_temp).mm(H.t())+R0))             # size is n*Hn

            #     Xkk=Xyuce_temp+Kk.mm(Zyuce[i,j-1]-H.mm(Xyuce_temp))
            #     Pkk=Pyuce_temp-Kk.mm(H).mm(Pyuce_temp)

        
        # DI I CI SHIYAN
        result[i]=(X[i,-1]-Xkk).mm((X[i,-1]-Xkk).t())
        resultPkkjy[i]=Pkk
        montecarlo[i]=torch.dist(torch.mean(result[:i+1],0),resultPkkjy[i],p=2)/torch.dist(resultPkkjy[i],torch.zeros(n,n),p=2)


    plt.subplot(211)
    plt.plot(X[i,::20,0,0].numpy(),X[i,::20,1,0].numpy(),'o-')
    plt.plot(Xkk_save[i,::20,0,0].numpy(),Xkk_save[i,::20,1,0].numpy(),'o-')
    # plt.plot(torch.sum((Zyuce[i,:,:2,0]-Xyuce[i,:,:2,0])*(Zyuce[i,:,:2,0]-Xyuce[i,:,:2,0]),dim=1)[:200].numpy(),'o-')

    plt.subplot(212)
    plt.plot(montecarlo.numpy(),'o-')
    plt.show()
    # print(montecarlo[-1])
    # # print(montecarlo[0])
    # for i in range(50):
    #     print(montecarlo[i])
        




        

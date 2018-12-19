import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.distributions.normal as normal

if __name__ == '__main__':

    S0=True     # WHEN S0=0 IS TRUE
    n=4        # length of X0
    Hn=2
    expnumb=100  # expr numb
    group=500   # group number

    ########################################
    X0ba=torch.tensor([[0.0],[0.0],[1.0],[1.0]])
    P=80.0
    P0=torch.eye(n)*P
    Q=0.05
    Q0=torch.eye(n)*Q
    R=0.01
    R0=torch.eye(Hn)*R
    F=torch.tensor([[1.0,0.0,1.0,0.0],[0.0,1.0,0.0,1.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
    G=torch.tensor([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
    # H=torch.tensor([[1.0,0.0],[0.0,0.0],[0.0,1.0],[0.0,0.0]])
    H=torch.tensor([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]])
    ########################################
    nX0=normal.Normal(X0ba, P)
    if S0:
        nWk=normal.Normal(torch.zeros(n,1), Q)
        nVk=normal.Normal(torch.zeros(Hn,1), R)
    else:
        pass
    ########################################
    #start
    X=torch.zeros(expnumb,group,n,1)
    Xyuce=torch.zeros(expnumb,group,n,1)
    Zyuce=torch.zeros(expnumb,group,Hn,1)
    # Xkk=torch.zeros(expnumb,group,n,n)
    # Pkk=torch.zeros(expnumb,group,n,n)
    result=torch.zeros(expnumb,n,n)
    resultPkkjy=torch.zeros(expnumb,n,n)
    montecarlo=torch.zeros(expnumb)
    ########################################
    for i in range(expnumb):
        for j in range(group):

            Wk=nWk.sample()
            Vk=nVk.sample()

            if j==0:

                X[i,j]=F.mm(nX0.sample())+G.mm(Wk)             # X1 by X0 n*1
                Xyuce[i,j]=F.mm(X0ba)+G.mm(Wk)                 # X1 by X0
                Zyuce[i,j]=H.mm(Xyuce[i,j])+Vk                 # Z1 by X1

                # measurement update, X00 and P00 (need Z0 by X0)
                Kk=P0.mm(H.t()).mm(torch.inverse(H.mm(P0).mm(H.t())+R0))  # size is n*Hn
                Z0=H.mm(X0ba)+Vk                                          # size is Hn*1

                Xkk=X0ba+Kk.mm(Z0-H.mm(X0ba))                             # size is n*1
                Pkk=P0-Kk.mm(H).mm(P0)                                    # size is n*n

            else:
                
                X[i,j]=F.mm(X[i,j-1])+G.mm(Wk)                       # X2 by X1 n*1
                Xyuce[i,j]=F.mm(Xyuce[i,j-1])+G.mm(Wk)               # X2 by X1 n*1
                Zyuce[i,j]=H.mm(Xyuce[i,j])+Vk                       # Z2 by X2 Hn*1

                # time update
                Xyuce_temp=F.mm(Xkk)                                 # size is n*1
                Pyuce_temp=F.mm(Pkk).mm(F.t())+G.mm(Q0).mm(G.t())    # size is n*n

                # measurement update X11 and P11 (need Z1)
                Kk=Pyuce_temp.mm(H.t()).mm(torch.inverse(H.mm(Pyuce_temp).mm(H.t())+R0))             # size is n*Hn

                Xkk=Xyuce_temp+Kk.mm(Zyuce[i,j-1]-H.mm(Xyuce_temp))
                Pkk=Pyuce_temp-Kk.mm(H).mm(Pyuce_temp)

        
        # DI I CI SHIYAN
        result[i]=(X[i,-1]-F.mm(Xkk)).mm((X[i,-1]-F.mm(Xkk)).t())
        resultPkkjy[i]=F.mm(Pkk).mm(F.t())+G.mm(Q0).mm(G.t())
        montecarlo[i]=torch.dist(torch.mean(result[:i+1],0),resultPkkjy[i],p=1)/torch.dist(resultPkkjy[i],torch.zeros(n,n),p=1)


    # plt.subplot(211)
    # plt.plot(X[2,:,:2,0][:,0].numpy(),X[2,:,:2,0][:,1].numpy(),'o-')
    # plt.plot(Xyuce[2,:,:2,0][:,0].numpy(),Xyuce[2,:,:2,0][:,1].numpy(),'o-')

    # plt.subplot(212)
    plt.plot(montecarlo.numpy())
    plt.show()
    # print(montecarlo[-1])
    # # print(montecarlo[0])
    # for i in range(10):
    #     print(montecarlo[i])
        




        

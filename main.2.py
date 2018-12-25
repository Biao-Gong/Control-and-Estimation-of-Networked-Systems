import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.distributions.normal as normal

if __name__ == '__main__':

    S0=True     # WHEN S0=0 IS TRUE
    n=4        # length of X0
    Hn=2
    expnumb=50  # expr numb
    group=1000   # group number

    ########################################
    X0ba=torch.tensor([[0.0],[0.0],[3.0],[3.0]])
    P=1.0
    P0=torch.eye(n)*P
    Q=0.5
    Q0=torch.eye(n)*Q
    R=1.1
    R0=torch.eye(Hn)*R
    TT=0.001
    F=torch.tensor([[1.0,0.0,TT,0.0],[0.0,1.0,0.0,TT],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
    G=torch.tensor([[0.5*TT*TT,0.0,0.0,0.0],[0.0,0.5*TT*TT,0.0,0.0],[0.0,0.0,TT,0.0],[0.0,0.0,0.0,TT]])
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
        montecarlo[i]=torch.dist(torch.mean(result[:i+1],0)/10000.0,resultPkkjy[i],p=1)/torch.dist(resultPkkjy[i],torch.zeros(n,n),p=1)


    plt.subplot(211)
    # plt.plot(Xyuce[i,:,0,0].numpy(),Xyuce[i,:,1,0].numpy(),'o-')
    # plt.plot(Zyuce[i,:,0,0].numpy(),Zyuce[i,:,1,0].numpy(),'o-')
    plt.plot(torch.sum((Zyuce[i,:,:2,0]-Xyuce[i,:,:2,0])*(Zyuce[i,:,:2,0]-Xyuce[i,:,:2,0]),dim=1)[:200].numpy(),'o-')

    plt.subplot(212)
    plt.plot(montecarlo.numpy(),'o-')
    plt.show()
    # print(montecarlo[-1])
    # # print(montecarlo[0])
    # for i in range(10):
    #     print(montecarlo[i])
        




        

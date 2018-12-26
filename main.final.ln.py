import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.distributions.normal as normal
import math

if __name__ == '__main__':

    # S0=False         # WHEN S0=0 IS TRUE
    n=4             # length of X0
    Hn=2
    expnumb=1000    # expr numb
    group=50       # group number

    ########################################
    X0ba=torch.tensor([[0.0],[0.0],[2.0],[2.0]])
    P=1.5
    P0=torch.eye(n)*P
    Q=0.1
    Q0=torch.eye(n)*Q
    R=0.12
    R0=torch.eye(Hn)*R
    Ts=0.0001
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
    L=torch.tensor([[1.0,0.0,0.0,0.0,0.0,0.0],
                    [1.0,2.0,0.0,0.0,0.0,0.0],
                    [1.0,2.0,3.0,0.0,0.0,0.0],
                    [1.0,2.0,3.0,4.0,0.0,0.0],
                    [1.0,2.0,3.0,4.0,2.0,0.0],
                    [1.0,2.0,3.0,4.0,1.0,3.0]])
    QRmat=torch.tensor([[Q,0.0,0.0,0.0,0.0,0.0],
                        [0.0,Q,0.0,0.0,0.0,0.0],
                        [0.0,0.0,Q,0.0,0.0,0.0],
                        [0.0,0.0,0.0,Q,0.0,0.0],
                        [0.0,0.0,0.0,0.0,R,0.0],
                        [0.0,0.0,0.0,0.0,0.0,R]])        
    S0=torch.tensor([[0.1000,0.1000],
                    [0.5000,0.5000],
                    [1.4000,1.4000],
                    [3.0000,3.0000]])

    ########################################
    nX0=normal.Normal(X0ba, math.sqrt(P))
    nWk=normal.Normal(torch.zeros(n,1), math.sqrt(Q))
    nVk=normal.Normal(torch.zeros(Hn,1), math.sqrt(R))
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
        
        if i%100==0:
            print(i)

        Xkk=nX0.sample()
        Pkk=P0
        Xyuce[i,0]=X0ba                                # X0yuce
        Zyuce[i,0]=H.mm(X0ba)+nVk.sample()
        
        for j in range(group-1):

            Wk=nWk.sample()
            Vk=nVk.sample()
            tempmat=L.mm(torch.cat((Wk.reshape(-1),Vk.reshape(-1))).reshape(-1,1))
            Wk=tempmat[:n,0].reshape(-1,1)
            Vk=tempmat[n:Hn+n,0].reshape(-1,1)


            # function
            Xyuce[i,j+1]=F.mm(Xyuce[i,j])+G.mm(Wk)     # X1 by X0
            Zyuce[i,j+1]=H.mm(Xyuce[i,j+1])+Vk         # Z1 by X1            

            # kalman
            Kk=(F.mm(Pkk).mm(H.t())+G.mm(S0)).mm(torch.inverse(H.mm(Pkk).mm(H.t())+R0))
            Xkk=F.mm(Xkk)+Kk.mm(Zyuce[i,j]-H.mm(Xkk))  # size is n*1
            Pkk=(F-Kk.mm(H)).mm(Pkk).mm((F-Kk.mm(H)).t())+\
            G.mm(Q0).mm(G.t())+\
            Kk.mm(R0).mm(Kk.t())-\
            G.mm(S0).mm(Kk.t())-\
            Kk.mm(S0.t()).mm(G.t())
            # size is n*n

            # save
            Xkk_save[i,j]=Xkk
            Pkk_save[i,j]=Pkk

        
        # DI I CI SHIYAN
        result[i]=(Xyuce[i,-1]-Xkk).mm((Xyuce[i,-1]-Xkk).t())
        resultPkkjy[i]=Pkk
        montecarlo[i]=torch.dist(torch.mean(result[:i+1],0),resultPkkjy[i],p=2)/torch.dist(resultPkkjy[i],torch.zeros(n,n),p=2)

    ########################################
    # plot
    # montecarlo
    plt.plot(montecarlo.numpy()[::10],'o-')
    plt.show()





        

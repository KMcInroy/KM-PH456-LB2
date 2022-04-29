import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator,MT19937

def ISMCIntegrator(F,W,Llim, Ulim,N,D): #Importance Sampling Monte-Carlo integrator
### Inputs: F-Integrand, Llim-lower limit of integral (array), ###
### Ulim- upper limit of integral (array), N- number of points to sample (int) ###
### The integrand should be written such that, for a function F(x,y,z): ###
### x=X[0],y=X[1],z[2] etc. ###
   ### Initialising indices and arrays ###
   i=1
   j=0
   c=0
   d=len(Ulim)
   X=np.empty(N,dtype=list)
   XArr=np.empty(N+1)
   FEvArr=np.empty(N,dtype=list)
   WEvArr=np.empty(N,dtype=list)
   StdDevArr=np.empty(N)
   StdDevArr2=np.empty(N)
   Success=0
   Fail=0
   ### Random Numbers, scaled to interval, d random numbers for each step ###
   XArr[0]=np.random.uniform(Llim[c],Ulim[c])
   ### Performing random walk across domain then evaluating F and W at each ###
   ### accepted point ###
   while i<N:
       XArr[i],Success,Fail=Metropolis(W,XArr[i-1],Success,Fail,D)
       FEvArr[i]=F(XArr[i])
       WEvArr[i]=W(XArr[i])
       StdDevArr[i]=(FEvArr[i]/W(XArr[i]))
       StdDevArr2[i]=(FEvArr[i]/W(XArr[i]))**2
       i+=1
   plt.hist(XArr, bins=np.linspace(-1,1,10))
   print(Success)
   print(Fail)
   ### Drop first 10% of results for better accuracy ###
   S=N//10
   FEvArr=FEvArr[S:]
   WEvArr=WEvArr[S:]
   FEvArr2=FEvArr[S:]
   WEvArr2=WEvArr[S:]
   ### Calculate integral and errors ###
   Integral=sum(FEvArr/WEvArr)/(N-S)
   StdDev=np.sqrt((np.mean(StdDevArr2)-np.mean(StdDevArr)**2)/N)
   print(Integral)
   
   
   return Integral,StdDev

    i=0
    d=len(Ulim)
    Diff=np.empty(d)
    while i<d:
        Diff[i]=Ulim[i]-Llim[i]
        i+=1
    return np.prod(Diff)
### Definition of Functions ###
def F5a(X):
    Y=X**2
    if X==0:
        E=2
    else:
        E=2*np.exp(-1*Y)
    return E

def W5a(X):
    return (1/(2-2*np.exp(-10)))*np.exp(-abs(X))

def F5b(X):
    return 1.5*np.sin(X)

def W5b(X):
    return (6/(np.pi**3))*X*(np.pi-X)

### Defining metropolis algorithm for a random walk ###
def Metropolis(W,X,S,F,d):
    ### Trial increment limit ###
    di=np.random.uniform(-d,d) 
    ### Trial step ###
    Xt=X+di
    w=W(Xt)/W(X)
    ### Acceptance criteria ###
    if w>=1:
        S +=1
        X=Xt
    else:
        r=np.random.uniform(0,1)
        if r<=w:
            X=Xt
            S +=1
        else:
            X=X
            F+=1
    return X,S,F

### This is the same integrator as T1-T4, refer to prior documentation ###
### In an effort to compact this script ###
def MCIntegrator(F, Llim, Ulim,N): #Monte Carlo integrator
   i=0
   j=0
   c=0
   d=len(Ulim)
   X=np.empty(N)
   XjArr=np.empty(d)
   FEvArr=np.empty(N,dtype=list)
   FEvArr2=np.empty(N,dtype=list)
   diff=DiffLim(Ulim,Llim)    
   while j<N:
      XjArr=np.empty(d, dtype=list)
      while c<d:
           XjArr[c]=np.random.uniform(Llim[c],Ulim[c])
           c+=1   
      X[j]=XjArr   
      c=0 
      j+=1
   while i<N:    
       FEvArr[i]=F(X[i])    
       FEvArr2[i]=FEvArr[i]**2    
       i+=1
   Integral=diff*sum(FEvArr)/N
   print(Integral)
   StdDev=np.sqrt((np.mean(FEvArr2)-np.mean(FEvArr)**2)/N)
   Errs=StdDev
   return Integral, Errs

### Calculations ###
Ia,Erra=ISMCIntegrator(F5a,W5a,[-10],[10],1000000,3.5)
Ib,Errb=ISMCIntegrator(F5b,W5b,[0],[np.pi],1000000,2.3)
IUSa,ErrUSa=MCIntegrator(F5a, [-10], [10], 100000)
IUSb,ErrUSb=MCIntegrator(F5b, [0], [np.pi], 10000000)
print("IS Method 5a",Ia,Erra)
print("IS Method 5b",Ib,Errb)
print("US Method 5a",IUSa,ErrUSa)
print("US Method 5b",IUSb,ErrUSb)

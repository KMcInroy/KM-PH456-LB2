import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator,MT19937

def MCIntegrator(F, Llim, Ulim,N): #Monte-Carlo integrator
### Inputs: F-Integrand, Llim-lower limit of integral (array), ###
### Ulim- upper limit of integral (array), N- number of points to sample (int) ###
### The integrand should be written such that, for a function F(x,y,z): ###
### x=X[0],y=X[1],z[2] etc. ###
   ### Initialising indices and arrays ###
   i=0
   j=0
   c=0
   d=len(Ulim)
   X=np.empty(N,dtype=list)
   XjArr=np.empty(d)
   FEvArr=np.empty(N,dtype=list)
   FEvArr2=np.empty(N,dtype=list)
   ### Calculating (b-a) for the integration ###
   diff=DiffLim(Ulim,Llim)

       
   while j<N:
      ### Redefine to empty for each dimension ###
      XjArr=np.empty(d, dtype=list)
      ### Random Numbers, scaled to interval, d random numbers for each step ###
      while c<d:
           XjArr[c]=np.random.uniform(Llim[c],Ulim[c])
           c+=1
      X[j]=XjArr
   
      c=0 
      j+=1
   
   while i<N:
       ### Evaluating the integrand and its square at each random point ###
       ### within the domain ###
       
       FEvArr[i]=F(X[i])
       
       FEvArr2[i]=FEvArr[i]**2
       
       i+=1
   
   ### Evaluating the integral and the associated errors ###
   Integral=diff*sum(FEvArr)/N
   #print(Integral)
   
   StdDev=np.sqrt((np.mean(FEvArr2)-np.mean(FEvArr)**2)/N)
   Errs=StdDev
   return Integral, Errs



def DiffLim(Ulim, Llim):
    ### Calculates the difference in the limits for a d-dimensional integral ###
    i=0
    d=len(Ulim)
    Diff=np.empty(d)
    while i<d:
        Diff[i]=Ulim[i]-Llim[i]
        i+=1
    return np.prod(Diff)

### Function Definitions ###
def F1(X):
    return 2

def F2(X):
    return -X[0]

def F3(X):
    return X[0]**2

def F4(X):
    return X[0]*X[1]+X[0]

def FStep(X):
    #print(X)
    i=0
    d=len(X)
    R2Arr=np.zeros(len(X))
    while i < d:
        
        R2Arr[i]=X[i]**2
        i+=1
    r=np.sqrt(sum(R2Arr))
    if r <= 2:
        F=1
    if r > 2:
        F=0
    return F

def F9D(X):
    #print(X)
    A=np.array([X[0],X[1],X[2]])
    B=np.array([X[3],X[4],X[5]])
    C=np.array([X[6],X[7],X[8]])
    return 1/abs(np.dot((A+B),C))

### Perform Calculations ###
Int1=MCIntegrator(F1,[0],[1],100000)
Int2=MCIntegrator(F2,[0],[1],100000)
Int3=MCIntegrator(F3,[-2],[2],100000)
Int4=MCIntegrator(F4,[0,0],[1,1],100000)
IntStep1=MCIntegrator(FStep,[-2,-2,-2],[2,2,2],10000000)
IntStep2=MCIntegrator(FStep,[-2,-2,-2,-2,-2],[2,2,2,2,2],10000000)
Int9D=MCIntegrator(F9D,[0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1],10000000)
print("I9D",Int9D)
print("I1",Int1)
print("I2",Int2)
print("I3",Int3)
print("I4",Int4)
print("3-Sphere Volume",IntStep1)
print("5-Sphere volume",IntStep2)
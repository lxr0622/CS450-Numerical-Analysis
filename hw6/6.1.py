import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

def func(u):
    n=len(u)
    uu=np.insert(u,[0,n],[0,1])
    A=np.zeros(n*(n+2)).reshape(n,n+2)
    for i in range(n):
        A[i,i]=1
        A[i,i+1]=-2
        A[i,i+2]=1
    t=np.linspace(0,1,n+2)
    t = t[1:len(t)-1]
    return (n+1)**2*A@uu-10*u**3-3*u-t**2

def solve(n):
    u_guess=np.delete(np.linspace(0,1,n+2),[0,n+1])
    u=(so.root(func,u_guess)).x
    uu=np.insert(u,[0,n],[0,1])
    return uu

n_list=[1,3,7,15]
for i in n_list:
    plt.plot(np.linspace(0,1,i+2), solve(i), '--o' , label = ("n="+str(i)))
plt.xlabel('t')
plt.ylabel('u')
plt.title('Solving a Boundary Value Problem')
plt.legend()
        
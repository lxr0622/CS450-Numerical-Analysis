import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

#euler forward method
def euler(n):
    v=0.1
    dx=1.0/(n+1)
    L=-v/(dx**2)*sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).toarray()
    f=np.ones(n)
    u=np.zeros(n)
    t=0
    dt=0.01
    u_list=[]
    i=0
    while t<10:
        u=u+dt*(L@u+f)
        t=t+dt
        if i%10==0:
            u_list.append(u)
        i=i+1
    j=0
    x_list=[]
    while j<n:
        x_list.append(j*dx)
        j=j+1
    return x_list,u_list

#plot
#n=15
x_list15,u_list15=euler(15)
plt.figure()
for i in range(len(u_list15)):
    plt.plot(x_list15,u_list15[i])
plt.title("u_j vs x_j for n=15")
plt.xlabel('x_j')
plt.ylabel('u_j')
print("the norm of u(t=10) is: "+str(np.linalg.norm(u_list15[-1],np.inf)))
#n=30
x_list30,u_list30=euler(30)
plt.figure()
for i in range(len(u_list30)):
    plt.plot(x_list30,u_list30[i])
plt.title("u_j vs x_j for n=30")
plt.xlabel('x_j')
plt.ylabel('u_j')
print("when n=30, the result becomes unstable")
#largest stable n
print("When n is decreasing from n=30, I found that the result become stable when n=21. Thus, the largest n is 21")
x_list21,u_list21=euler(21)
x_list22,u_list22=euler(22)
plt.figure()
for i in range(len(u_list21)):
    plt.plot(x_list21,u_list21[i],'b')
    plt.plot(x_list22,u_list22[i],'r')
plt.title("u_j vs x_j for n=21(blue) and 22(red)")
plt.ylim(0,2)
plt.xlabel('x_j')
plt.ylabel('u_j')

        

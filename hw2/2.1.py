import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
import time
import math

#Returns the product (A x B x C)u
def kron_eval(A, B, C, u):
    ma, na = A.shape
    mb, nb = B.shape
    mc, nc = C.shape

    v = np.zeros((nc, mb, na))
    u = u.reshape((nc, nb, na))

    Bt = B.transpose()

    for k in range(na):
        v[:, :, k] = u[:, :, k] @ Bt

    v = v.reshape((nc, nb*na))
    v = C @ (v)
    
    v = v.reshape((mc*nb, nc))
    v = v @ (A.transpose())
    
    v = v.reshape((ma*mb*mc, 1))
    return v

#Define Ax
def Define_Ax(N):
    h=1./(N+1)
    Ax=1./(h**2)*sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N)).toarray()
    return Ax

#Define A
def Define_A(Ax,I):
    A=(sp.kron(sp.kron(I,I),Ax)+sp.kron(sp.kron(I,Ax),I)+sp.kron(sp.kron(Ax,I),I)).toarray()
    return A

#computational time for a slow way
def slow_time(N,f):
    Ax=Define_Ax(N)
    I=np.identity(N)
    A=Define_A(Ax,I)
    
    start_time=time.time()
    la.spsolve(A,f)
    end_time=time.time()
    return (end_time-start_time)

#fast solve way
def fast_solve(Ax, f):
    #identity matrix
    N,N=Ax.shape
    I=sp.identity(N)
    #eigen values and eigen vectors
    Evalue=np.zeros(shape=(N,N))
    Evector=np.zeros(shape=(N,N))
    h=1./(N+1)
    for i in range(N):
        Evalue[i,i]=2./(h**2)*(1-math.cos(math.pi*h*(i+1)))
        for j in range(N):
             Evector[i,j]=math.sqrt(2*h)*math.sin(math.pi*h*(i+1)*(j+1))
    #calcualte u
    u1=kron_eval(Evector.T,Evector.T,Evector.T,f).reshape(N**3,1)
    u2=(sp.kron(sp.kron(I,I),Evalue)+sp.kron(sp.kron(I,Evalue),I)+sp.kron(sp.kron(Evalue,I),I))
    for i in range(N**3):
        u1[i]=u1[i]/u2[i,i]
#    u1=np.divide(u1,u2)
    u3=kron_eval(Evector,Evector,Evector,u1)
#    u1=sp.kron(sp.kron(Evector,Evector),Evector)
#    u2=np.dot(u1,la.inv(sp.kron(sp.kron(I,I),Evalue)+sp.kron(sp.kron(I,Evalue),I)+sp.kron(sp.kron(Evalue,I),I)))
#    u3=np.dot(u2,kron_eval(Evector.transpose(),Evector.transpose(),Evector.transpose(),f))
#    u1=kron_eval(np.transpose(Evector),np.transpose(Evector),np.transpose(Evector),f)
#    u2=np.dot(la.inv(sp.kron(sp.kron(I,I),Evalue)+sp.kron(sp.kron(I,Evalue),I)+sp.kron(sp.kron(Evalue,I),I)),u1)
#    u3=np.dot(sp.kron(sp.kron(Evector,Evector),Evector),u2)
    return u3

##fast solve way
#def fast_solve(Ax, f):
#    #eigen value w and eigen vector v
#    Evalue,Evector = np.linalg.eigh(Ax)
#    Evalue=sp.diags(Evalue)
#    #identity matrix
#    N,M = Ax.shape
#    I = np.identity(N)
#    #calcualte u
##    u1=sp.kron(sp.kron(Evector,Evector),Evector)
##    u2=la.inv(sp.kron(sp.kron(I,I),Evalue)+sp.kron(sp.kron(I,Evalue),I)+sp.kron(sp.kron(Evalue,I),I))
##    u3=u1*u2*kron_eval(Evector.transpose(),Evector.transpose(),Evector.transpose(),f)
#    u1=kron_eval(Evector.transpose(),Evector.transpose(),Evector.transpose(),f)
#    u2=la.inv(sp.kron(sp.kron(I,I),Evalue)+sp.kron(sp.kron(I,Evalue),I)+sp.kron(sp.kron(Evalue,I),I))*u1
#    u3=kron_eval(Evector,Evector,Evector,u2)
#    return u3


#computational time for a fast way
def fast_time(N,f):
    Ax=Define_Ax(N)
    
    start_time=time.time()
    fast_solve(Ax,f)
    end_time=time.time()
    return (end_time-start_time)

#Time for each N
T_slow=[]
T_fast=[]
n_arr=[]
for N in range(5,20):
    n=N**3
    #elements of f follows the standard normal distribution
    f=np.random.rand(n)
    T_slow.append(slow_time(N,f))
    T_fast.append(fast_time(N,f))
    n_arr.append(n)

#plot
spsolve,=plt.plot(n_arr,T_slow,'b-',label='spsolve')
fastsolve,=plt.plot(n_arr,T_fast,'r-',label='fastsolve')
plt.xlabel('n')
plt.ylabel('Time(s)')
plt.legend(handles=[spsolve,fastsolve])
plt.title('Comparison of tensor solves')




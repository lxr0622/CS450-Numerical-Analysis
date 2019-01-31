import numpy as np
import numpy.linalg as npla
import scipy.sparse as sp
import matplotlib.pyplot as plt

def Diri(n):
    h=1/(n+1)
    A=sp.diags([-1,2,-1],[-1,0,1],shape=(n,n)).toarray()
    A=A/h
    B=sp.diags([1.0/6,2.0/3,1.0/6],[-1,0,1],shape=(n,n)).toarray()
    B=B*h
    return npla.eig((npla.inv(B))@A)[0]

def Neum(n):
    h=1/(n+1)
    A=sp.diags([-1,2,-1],[-1,0,1],shape=(n+1,n+1)).toarray()
    A[-1,-1]/=2
    A=A/h
    B=sp.diags([1.0/6,2.0/3,1.0/6],[-1,0,1],shape=(n+1,n+1)).toarray()
    B[-1,-1]/=2
    B=B*h
    return npla.eig((npla.inv(B))@A)[0]

n_list=np.power(2,np.arange(1,10))
Diri_list=[]
Neum_list=[]
i=0

for i in n_list:
    Diri_list.append(min(Diri(i)))
    Neum_list.append(min(Neum(i)))
    i=i+1
    
Diri_err=np.abs(Diri_list-((np.pi)**2)*np.ones(9))
Neum_err=np.abs(Neum_list-(0.25*(np.pi)**2)*np.ones(9))

plt.figure()
plt.title("absolute error vs n")
plt.loglog(n_list, Diri_err, label = "Dirichlet")
plt.loglog(n_list, Neum_err, label = "Neumann")
plt.loglog(n_list, 1/(n_list**2), label = "O(n^2)")
plt.xlabel('n')
plt.ylabel('error')
plt.legend()

print("For Neumann method, I increase the matrix size to n+1 and take the half value at lower right corner for A matrix and B matrix")


#plt.figure()
#plt.title("minimum eigenvalue vs n")
#plt.loglog(n_list, Diri_list, label = "Dirichlet")
#plt.loglog(n_list, Neum_list, label = "Neumann")
#plt.xlabel('n')
#plt.ylabel('eigenvalue')
#plt.legend()
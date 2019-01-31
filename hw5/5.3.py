import numpy as np
import matplotlib.pyplot as plt

n_list=np.arange(2,51)
err1=[]
err2=[]
for n in range(2,51):
    # approximate with uniformly distributed points
    x1=np.linspace(-1,1,n).reshape((n,1))
    c1=np.ones((n,1))
    D1=np.zeros((n,n))
    f1=np.zeros((n,n))
    
    # approximate with Gauss-Legendre quadrature points
    x2,w=np.polynomial.legendre.leggauss(n)
    c2=np.ones((n,1))
    D2=np.zeros((n,n))
    f2=np.zeros((n,n))
    
    # generate derivative matrix D
    for i in range(n):
        for j in range(n):
            if j !=i:
                c1[i]=c1[i]*(x1[i]-x1[j])
                c2[i]=c2[i]*(x2[i]-x2[j])
        c1[i]=1.0/c1[i]
        c2[i]=1.0/c2[i]
        
    for j in range(n):
        for i in range(n):
            D1[i,j]=x1[i]-x1[j]
            D2[i,j]=x2[i]-x2[j]
        D1[j,j]=1
        D2[j,j]=1
    D1=1./D1
    D2=1./D2
    
    for i in range(n):
        D1[i,i]=0
        D1[i,i]=np.sum(D1[i,:])
        D2[i,i]=0
        D2[i,i]=np.sum(D2[i,:])
    
    for j in range(n):
        for i in range(n):
            if i !=j:
                D1[i,j]=c1[j]/(c1[i]*(x1[i]-x1[j]))
                D2[i,j]=c2[j]/(c2[i]*(x2[i]-x2[j]))

    #error
    err1.append(np.max(np.abs((D1@np.cos(15*x1)-(-15*np.sin(15*x1))))))
    err2.append(np.max(np.abs((D2@np.cos(15*x2)-(-15*np.sin(15*x2))))))

#plot
plt.figure()
plt.semilogy(n_list, err1)
plt.xlabel('number of points')
plt.ylabel('Error')
plt.title('Accuracy for approximation with uniform points')

plt.figure()
plt.semilogy(n_list, err1,label='uniform points')
plt.semilogy(n_list, err2,label='Gauss-Legendre quadrature points')
plt.xlabel('number of points')
plt.ylabel('Error')
plt.legend()
plt.title('Comparison of Accuracy')

import numpy as np
import matplotlib.pyplot as plt

#computing function
def compute(x,n):
    if n==0:
        return ((x**2+2)/3)
    elif n==1:
        return ((3*x-2)**0.5)
    elif n==2:
        return (3-2/x)
    elif n==3:
        return ((x**2-2)/(2*x-3))
    
x_mat = np.zeros((10, 4))
x0 = 2.2
for j in range(4):
    for i in range(10):
        if i==0:
            x_mat[i,j] = compute(x0, j)
        else: 
            x_mat[i,j] = compute(x_mat[i-1,j], j)
            

#plot
n_list=np.arange(1,11)
plt.figure()
g1,=plt.semilogy(n_list,abs(x_mat[:,0]-2)/2,label='g1=(x**2+2)/3')
g2,=plt.semilogy(n_list,abs(x_mat[:,1]-2)/2,label='g2=(3*x-2)**0.5')
g3,=plt.semilogy(n_list,abs(x_mat[:,2]-2)/2,label='g3=3-2/x')
g4,=plt.semilogy(n_list,abs(x_mat[:,3]-2)/2,label='g4=(x**2-2)/(2*x-3)')
plt.xlabel('iterations')
plt.ylabel('relative error')
plt.legend(handles=[g1,g2,g3,g4])
plt.title('Comparison of fixed point method')
# g1 is divergent

    
plt.figure()
g2,=plt.semilogy(n_list,abs(x_mat[:,1]-2)/2,label='g2=(3*x-2)**0.5')
g3,=plt.semilogy(n_list,abs(x_mat[:,2]-2)/2,label='g3=3-2/x')
g4,=plt.semilogy(n_list,abs(x_mat[:,3]-2)/2,label='g4=(x**2-2)/(2*x-3)')
plt.xlabel('iterations')
plt.ylabel('relative error')
plt.legend(handles=[g2,g3,g4])
plt.title('Comparison of fixed point method (exclude divergent method)')
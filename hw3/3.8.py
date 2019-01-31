import numpy as np
import matplotlib.pyplot as plt

# Newton's method
x_true=np.array([[0],[1]])
x0=np.array([[-0.5],[1.4]])
x=x0
J=np.zeros((2,2))
N=0
N_error=[]

while np.linalg.norm(x-x_true)>=np.finfo(float).eps:
    J[0][0]=x[1]**3-7
    J[0][1]=(3*x[1]**2)*(x[0]+3)
    J[1][0]=x[1]*np.exp(x[0])*np.cos(x[1]*np.exp(x[0])-1)
    J[1][1]=np.exp(x[0])*np.cos(x[1]*np.exp(x[0])-1)
    x1=np.asscalar((x[0]+3)*(x[1]**3-7)+18)
    x2=np.asscalar(np.sin(x[1]*np.exp(x[0])-1))
    f=np.array([[x1],[x2]])
    s=np.linalg.solve(J,-f)
    x=x+s
    N=N+1
    N_error.append(np.linalg.norm(x-x_true))



# Broyden's method
x_true=np.array([[0],[1]])
x0=np.array([[-0.5],[1.4]])
x=x0
B=0
B_error=[]
J=np.zeros((2,2))
J[0][0]=x[1]**3-7
J[0][1]=(3*x[1]**2)*(x[0]+3)
J[1][0]=x[1]*np.exp(x[0])*np.cos(x[1]*np.exp(x[0])-1)
J[1][1]=np.exp(x[0])*np.cos(x[1]*np.exp(x[0])-1)

while np.linalg.norm(x-x_true)>=np.finfo(float).eps:
    x1=np.asscalar((x[0]+3)*(x[1]**3-7)+18)
    x2=np.asscalar(np.sin(x[1]*np.exp(x[0])-1))
    f=np.array([[x1],[x2]])
    B_error.append(np.linalg.norm(x-x_true))
    s=np.linalg.solve(J,-f)
    # Xk+1
    x=x+s
    x1=np.asscalar((x[0]+3)*(x[1]**3-7)+18)
    x2=np.asscalar(np.sin(x[1]*np.exp(x[0])-1))
    y=np.array([[x1],[x2]])-f
    J=J+np.dot((y-np.dot(J,s)),np.transpose(s))/(np.dot(np.transpose(s),s))
    B=B+1

print('Newton:',N)
print('Broyden:',B)

#plot
N_list=np.arange(1,N+1)
B_list=np.arange(1,B+1)
plt.figure()
Newton,=plt.semilogy(N_list,N_error,label='Newton')
Broyden,=plt.semilogy(B_list,B_error,label='Broyden')
plt.xlabel('iterations')
plt.ylabel('error')
plt.legend(handles=[Newton,Broyden])
plt.title('Implementing Newton and Broyden')
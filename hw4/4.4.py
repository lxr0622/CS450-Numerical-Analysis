import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import time
start_time = time.time()

# uniformly-spaced points T
def T(m):
    x=np.zeros((m,1))
    h=2/m
    for i in range(1,m+1):
        x[i-1,0]=-1-h/2+i*h
    return x
    

# uniformly-spaced nodes 
def X(n):
    x=np.linspace(-1,1,n).reshape((n,1))
    return x

# vector for q
def Q_vec(t,x):
    m=np.shape(t)[0]      
    n=np.shape(x)[0]
    q=np.ones((m,1))  
    for j in range(0,n):
            q=q*(t-x[j,0])
    return q

# Jacobi matrix 
def Jacobi(t,R,x):
    m=np.shape(t)[0]
    n=np.shape(x)[0]
    M=np.zeros((m,n))
    l=np.zeros((m,1))
    for i in range(n):
        l=(x[i,0]-t)
        for j in range(m):
            if l[j,0]!=0:
                M[j,i]=R[j,0]/l[j,0]
            else:
                 break
    return M
   
# Gauss-Newton
def GN(t,x0):
    x=x0
    for k in range(50):
        R=Q_vec(t,x)
        J=Jacobi(t,R,x)
        [q,r]=np.linalg.qr(J)
        s=np.linalg.solve(r,-q.T@R)
        if la.norm(s)<=10**-15:
            break
        x=x+s
    return Q_vec(t,x)

# Levenberg-Marquardt
def LM(t,x0):
    x=x0
    n=np.shape(x)[0]
    for k in range(50):
        R=Q_vec(t,x)
        J=Jacobi(t,R,x)
        A=J.T@J+np.exp(-(k+1))*np.identity(n)
        b=-J.T@R
        s=np.linalg.solve(A,b)
        if la.norm(s)<=10**-15:
            break
        x=x+s
    return Q_vec(t,x)


#plot(q vs t by LM)
plt.figure()
o,=plt.plot(T(300),LM(T(300),X(40)),label='optimized')
u,=plt.plot(T(300),Q_vec(T(300),X(40)),label='uniformed')
plt.xlabel('t')
plt.ylabel('q')
plt.legend(handles=[u,o])
plt.title('q vs t by Levenberg-Marquardt')
print("the accuracy is", la.norm(LM(T(300),X(40))))

#plot(qnorm vs n)
LM_optimized_norm=[]
GN_optimized_norm=[]
uniformed_norm=[]
n_list=[]
for i in range(1,41):
    LM_optimized_norm.append(la.norm(LM(T(300),X(i))))
    GN_optimized_norm.append(la.norm(GN(T(300),X(i))))
    uniformed_norm.append(la.norm(Q_vec(T(300),X(i))))
    n_list.append(i)
    
plt.figure()
LM_O,=plt.semilogy(n_list,LM_optimized_norm,label='optimized by Levenberg-Marquardt')
GN_O,=plt.semilogy(n_list,GN_optimized_norm,label='optimized by Gauss-Newton')
U,=plt.semilogy(n_list,uniformed_norm,label='uniformed')
plt.xlabel('n')
plt.ylabel('q_norm')
plt.legend(handles=[U,GN_O,LM_O])
plt.title('q_norm vs n')   
print('Gauss-Newtwon method fail to converge when n reach 11.Levenberg-Marquardt converge much better and keep increasing accuracy when n increase' )
print("--- %s seconds ---" % (time.time() - start_time))
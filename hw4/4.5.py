import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

def ydata(x):
    t = 20*np.random.rand(200,1) #Generate random points on interval
    t = np.sort(t, axis=0) #Sort the points
    #Evaluate function at points
    y = x[0,0]*np.exp(-x[1,0]*t)*np.sin(x[2,0]*t-x[3,0])+x[4,0]
    return y, t

a = 0.3 + 2*(np.random.rand() - 0.5)/2
b = 0.1 + 2*(np.random.rand() - 0.5)/25
omega = 4 + 2*(np.random.rand() - 0.5)/2
phase = -1.0 + (2*(np.random.rand() - 0.5))/2
c = 1.0 + (2*(np.random.rand() - 0.5))/2
coeffs = np.array([a, b, omega, phase, c])
#coeffs = np.array([0.3, 0.1, 4.0, -1.0, 1.0])

coeffs = coeffs.reshape((5,1))
[y,t] = ydata(coeffs)

#print(y) #Your code can access these provided  n x 1 numpy arrays
#print(t)

#You can use this as your initial guess.
x = np.array([0.3, 0.1, 4.0, -1.0, 1.0])
x = x.reshape((5,1))

# residual
def Res(y,t,x):
    return (y-(x[0,0]*np.exp(-x[1,0]*t)*np.sin(x[2,0]*t+x[3,0])+x[4,0]))

# Jacobi matrix 
def Jacobi(t,x):
    c1=-np.exp(-x[1,0]*t)*np.sin(x[2,0]*t+x[3,0])
    c2=x[0,0]*t*np.exp(-x[1,0]*t)*np.sin(x[2,0]*t+x[3,0])
    c3=-x[0,0]*t*np.exp(-x[1,0]*t)*np.cos(x[2,0]*t+x[3,0])
    c4=-x[0,0]*np.exp(-x[1,0]*t)*np.cos(x[2,0]*t+x[3,0])
    c5=-np.ones((len(t),1))
    M=np.hstack((c1,c2,c3,c4,c5))
#    M=np.zeros((len(t),len(x)))
#    for i in range(len(t)):
#        M[i,0]=(-np.exp(-x[1]*t[i])*np.sin(x[2]*t[i]+x[3]))
#        M[i,1]=x[0]*t[i]*np.exp(-x[1]*t[i])*np.sin(x[2]*t[i]+x[3])
#        M[i,2]=-x[0]*t[i]*np.exp(-x[1]*t[i])*np.cos(x[2]*t[i]+x[3])
#        M[i,3]=-x[0]*np.exp(-x[1]*t[i])*np.cos(x[2]*t[i]+x[3])
#        M[i,4]=-1
    return M

# Levenberg-Marquardt
def LM(y,t,x0):
    x=x0
    r=Res(y,t,x)
    n=np.shape(x)[0]
    while la.norm(r)>10**(-6):
        J=Jacobi(t,x)
        A=J.T@J+(10**(-5))*np.identity(n)
        b=-J.T@r
        s=la.solve(A,b)
        x=x+s
        r=Res(y,t,x)
    return x

x=LM(y,t,x)
norm_r=la.norm(Res(y,t,x))

plt.figure()
plt.plot(np.linspace(0, 20, 200).T,(x[0,0]*np.exp(-x[1,0]*t)*np.sin(x[2,0]*t+x[3,0])+x[4,0]),label='model')
plt.plot(t,y,'ro',label='data')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('nolinear least square fit')

#def L_M(t,x,y):
#    f=x[0][0]*np.exp(-x[1][0]*t)*np.sin(x[2][0]*t+x[3][0])+x[4][0]
#    r=y-f
#    norm_r=la.norm(r,ord=2)
#    r_matrix=np.matrix(r)
#    while norm_r>10**(-6):
#        miu=np.dot(r.T,r)
#        A=np.dot(Jacobi(y,x).T,Jacobi(t,x))+miu*np.eye(len(x))
#        B=-np.dot(Jacobi(t,x).T,r)
#        sk=la.lstsq(A,B)[0]
#        #print('len of sk',sk)
#        x=sk+x
#        f=x[0][0]*np.exp(-x[1][0]*t)*np.sin(x[2][0]*t+x[3][0])+x[4][0]
#        r=y-f
#        norm_r=la.norm(r,ord=2)
#        r_matrix=np.matrix(r)
#    return x,norm_r
#t_m = np.linspace(0, 20, 100).T
#x,norm_r = L_M(t,x,y)
#plt.plot(t_m,x[0][0]*np.exp(-x[1][0]*t_m)*np.sin(x[2][0]*t_m+x[3][0])+x[4][0],'k-',label='model')
#plt.plot(t,y,'ro-',label='data')
#plt.legend()
#plt.show()


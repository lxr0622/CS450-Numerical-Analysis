import numpy as np
import scipy

x0=np.array([[0],[0],[1]])
A=np.array([[6,2,1],[2,3,1],[1,1,1]])

# compute by numpy
w,v=np.linalg.eig(A)

# compute by inverse iteration with shift
lu,piv=scipy.linalg.lu_factor(A-2*np.identity(3))
x=x0
for i in range(0,15):
    y=scipy.linalg.lu_solve((lu,piv),x)
    x=y/np.linalg.norm(y)
eigval=1/np.linalg.norm(y)+2
eigvec=np.array([x[0][0],x[1][0],x[2][0]])

diffval=abs(w[1]-eigval)/w[1]

print(w)
print(x)
print(eigval)
print(eigvec)

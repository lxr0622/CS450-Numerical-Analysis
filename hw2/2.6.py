import numpy as np
import scipy.sparse as sp

A = np.random.rand(4,2)
z = np.random.rand(4,1)


#weighted least square approximation(WFQA)
def WFQA(A,W,z):
    return (np.dot(A,np.dot(np.linalg.inv(np.dot(np.dot(A.transpose(),W),A)),np.dot(A.transpose(),z))))

#W matrix
W1=np.identity(4)
W2=np.diag([1,2,3,4])
W3=(5**2)*sp.diags([-1, 2, -1], [-1, 0, 1], shape=(4, 4)).toarray()

#calculate projection of b
y1=WFQA(A,W1,z)
y2=WFQA(A,W2,z)
y3=WFQA(A,W3,z)
print(y1)
print(y2)
print(y3)
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt

ratio=np.zeros(20)

for i in range(2,22):
    matrix=np.array(2*np.identity(i)-np.transpose(np.tri(i,i)))
    ratio[i-2] = npla.svd(matrix)[1][0]/npla.svd(matrix)[1][i-1]

size=np.arange(2,22)    
plt.figure()
plt.plot(size,ratio,label='sigma(max)/sigma(min)')
plt.xlabel('Size of Matrix')
plt.ylabel('Ratio of singular values')
plt.title('Eigenvalues and Conditioning')

print('When the size of matrix increase, the ratio of singular values grow exponentially, which means the matrix becomes more and more ill-conditioned')
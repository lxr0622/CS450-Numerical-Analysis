import numpy as np

# matrix A
A=np.array([(1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,1.,0.),(0.,0.,0.,1.),(1.,-1.,0.,0.),(1.,0.,-1.,0.),(1.,0.,0.,-1.),(0.,1.,-1.,0.),(0.,1.,0.,-1.),(0.,0.,1.,-1.)])
# vector b
b=np.array([2.95,1.74,-1.45,1.32,1.23,4.45,1.61,3.21,0.45,-2.75])
# solution to the least-squares system
x=np.linalg.lstsq(A, b)[0]
# relative error
rel_errors=np.zeros(4) 
for i in range(4):
    rel_errors[i]=abs((b[i]-x[i])/x[i])
print(rel_errors)
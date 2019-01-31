import numpy as np
import matplotlib.pyplot as plt

print('The eigen values of A are -1+0.5i,-1-0.5i,-4.In order to be stable, abs(1+lamda*dt)<1.Thus, dt<=0.5')

#matrix A
A=np.array([[0,0,-5],[1,0,-9.25],[0,1,-6]])

#smaller step
h1=0.49
y1=3*np.ones(3)
Y1=y1
norm1=np.zeros(41)
norm1[0]=np.linalg.norm(y1)
#larger step
h2=0.51
y2=3*np.ones(3)
Y2=y2
norm2=np.zeros(41)
norm2[0]=np.linalg.norm(y2)

for i in range(40):
    y1=y1+h1*A@y1
    y2=y2+h2*A@y2
    Y1=np.vstack((Y1,y1))
    Y2=np.vstack((Y2,y2))
    norm1[i+1]=np.linalg.norm(y1)
    norm2[i+1]=np.linalg.norm(y2)

    
t1=np.arange(0,0.49*41,0.49)
t2=np.arange(0,0.51*41,0.51)
#comparison for each component of two solutions
plt.figure()
plt.plot(t1, Y1[:,0],'--o',label='x_dt=0.49')
plt.plot(t1, Y1[:,1],'--o',label='y_dt=0.49')
plt.plot(t1, Y1[:,2],'--o',label='z_dt=0.49')
plt.plot(t2, Y2[:,0],label='x_dt=0.51')
plt.plot(t2, Y2[:,1],label='y_dt=0.51')
plt.plot(t2, Y2[:,2],label='z_dt=0.51')
plt.xlabel('t')
plt.ylabel('component of solution')
plt.legend()
plt.title('comparison for each component of two solutions')    
    
#comparison for norm of two solutions
plt.figure()
plt.plot(t1, norm1,label='norm of solution_dt=0.49')
plt.plot(t2, norm2,label='norm of solution_dt=0.51')
plt.xlabel('t')
plt.ylabel('norm of solution')
plt.legend()
plt.title('comparison for norm of two solutions')    
 
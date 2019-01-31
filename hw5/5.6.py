import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint   

#function,let x=y1, y=y2, z=y3
def func(w,t,c,d):
    x,y,z=w.tolist()
    return -c*x*y,c*x*y-d*y,d*y
#parameter, initial value, t range
t=np.linspace(0,1,101)
y0=[95,5,0]
c=1
d=5
#solution
sol=odeint(func,y0,t,args=(c,d))
y1=sol[100,:]
#plot
plt.figure()
plt.plot(t, sol[:,0],label='y1')
plt.plot(t, sol[:,1],label='y2')
plt.plot(t, sol[:,2],label='y3')
plt.xlabel('t')
plt.ylabel('solution')
plt.legend()
plt.title('Modeling Epidemics as ODEs')
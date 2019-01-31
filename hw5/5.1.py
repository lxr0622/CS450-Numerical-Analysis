import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return(4./(1.+x**2))

#number of intervals for first 4 methods
num = np.array([2**i for i in range(1,11)])

#midpoint trapezoid simpson 
mid=np.zeros(len(num))
trap=np.zeros(len(num))
simp=np.zeros(len(num))

for i in range(len(num)):
    x_mid=np.linspace(0+1/(num[i]*2),1-1/(num[i]*2),num[i])
    x_trap=np.linspace(0,1,num[i]+1)
    x_simp=np.linspace(0,1,num[i]+1)
    for j in range(num[i]):
        mid[i]+=1.0/num[i]*f(x_mid[j])
        
        a=x_trap[j]
        b=x_trap[j+1]
        trap[i]+=(b-a)/2*(f(a)+f(b))
    for k in range(int(num[i]/2)):
        c=x_simp[k*2]
        d=x_simp[k*2+1]
        e=x_simp[k*2+2]
        simp[i]+=1.0/(num[i]*3)*(f(c)+4*f(d)+f(e))
        
#romberg 
T=np.zeros((len(num),len(num)))
T[:,0]=trap.copy()
for i in range (1,len(num)):
    T[i:,i]=(4**i*T[i:,(i-1)]-T[(i-1):-1,(i-1)])/(4**i-1)
rom=np.diag(T)

#Gauss-Legendre quadrature
num_gauss= np.linspace(4,30,10)
gauss=np.zeros(len(num_gauss))
for i in range(len(num_gauss)):
    n,w=np.polynomial.legendre.leggauss(int(num_gauss[i]))
    n=(n+1)/2
    w=0.5*w
    gauss[i]=np.sum(w*f(n))


#error
mid_err = np.abs((mid - np.pi)/np.pi)
trap_err = np.abs((trap - np.pi)/np.pi)
simp_err = np.abs((simp - np.pi)/np.pi)
rom_err = np.abs((rom - np.pi)/np.pi)
gauss_err = np.abs((gauss - np.pi)/np.pi)

#plot
plt.figure()
plt.loglog(num, mid_err, '--o', label='Midpoint')
plt.loglog(num, trap_err, '--o', label='Trapezoid')
plt.loglog(num, simp_err, '--o', label='Simpson')
plt.loglog(num, rom_err, '--o', label='Romberg')
plt.loglog(num_gauss, gauss_err, '--o', label='Gauss')
plt.xlabel('Number of function evaluations')
plt.ylabel('Relative Error')
plt.title('Comparison of Accuracy for Different Quadrature Rules')
plt.legend()
    
print("to reach a given error tolerance (10^-16), number of evaluations: trapezoid>midpoint>simpson>romberg>gauss")

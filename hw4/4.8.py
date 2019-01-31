import scipy.interpolate as si
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

#Part1

#function1
def f1(x):
    return (1/(1+25*x**2))
#function2
def f2(x):
    return (np.exp(np.cos(x)))

#dense points
x=np.linspace(-1,1,120)
#uniform points
n=12
x_uniform=np.linspace(-1,1,n)
#chebyshev points
x_chebyshev=np.zeros(n)
for i in range(1,n+1):
    x_chebyshev[i-1]=np.cos((2*i - 1) * np.pi/(2*n))

# lagrange with uniform points
lu=si.lagrange(x_uniform,f1(x_uniform))
# cubic spines with uniform points
cs=si.CubicSpline(x_uniform,f1(x_uniform))
# lagrange with chebyshev points
lc=si.lagrange(x_chebyshev,f1(x_chebyshev))

#plot 
plt.figure()
o,=plt.plot(x,f1(x),label='original function with uniform points')
lu,=plt.plot(x,lu(x),label='lagrange with uniform points')
cs,=plt.plot(x,cs(x),label='cubic spines with uniform points')
lc,=plt.plot(x,lc(x),label='lagrange with chebyshev points')
plt.xlabel('x')
plt.ylabel('f')
plt.legend(handles=[o,lu,cs,lc],bbox_to_anchor=(0.6, 0.6))
plt.title('three interpolants and the original function')   
print('Lagrange with uniform points fail to converge; Compared with lagrange with chebyshev points, Cubic Spine converge better')

#Part 2

#error
lu1_list=[]
cs1_list=[]
lc1_list=[]
lu2_list=[]
cs2_list=[]
lc2_list=[]

for n in range(4,51):
    #Function1
    #dense points
    x1=np.linspace(-1,1,n*10)
    #uniform points
    x_uniform1=np.linspace(-1,1,n)
    #chebyshev points
    x_chebyshev1=np.zeros(n)
    for i in range(1,n+1):
        x_chebyshev1[i-1]=np.cos((2*i - 1) * np.pi/(2*n))
    # lagrange with uniform points
    lu1=si.lagrange(x_uniform1,f1(x_uniform1))
    # cubic spines with uniform points
    cs1=si.CubicSpline(x_uniform1,f1(x_uniform1))
    # lagrange with chebyshev points
    lc1=si.lagrange(x_chebyshev1,f1(x_chebyshev1))
    #append the norm of error
    lu1_list.append(la.norm(lu1(x1)-f1(x1)))
    cs1_list.append(la.norm(cs1(x1)-f1(x1)))
    lc1_list.append(la.norm(lc1(x1)-f1(x1)))
    
    #Function2
    #dense points
    x2=np.linspace(0,2*np.pi,n*10)
    #uniform points
    x_uniform2=np.linspace(0,2*np.pi,n)
    #chebyshev points
    x_chebyshev2=np.zeros(n)
    for i in range(1,n+1):
        x_chebyshev2[i-1]=(np.cos((2*i - 1) * np.pi/(2*n))+1)*np.pi
    # lagrange with uniform points
    lu2=si.lagrange(x_uniform2,f2(x_uniform2))
    # cubic spines with uniform points
    cs2=si.CubicSpline(x_uniform2,f2(x_uniform2))
    # lagrange with chebyshev points
    lc2=si.lagrange(x_chebyshev2,f2(x_chebyshev2))
    #append the norm of error
    lu2_list.append(la.norm(lu2(x2)-f2(x2)))
    cs2_list.append(la.norm(cs2(x2)-f2(x2)))
    lc2_list.append(la.norm(lc2(x2)-f2(x2)))

#plot
N_list=np.arange(4,51,1) 
#Function1
plt.figure()
lu1,=plt.semilogy(N_list,lu1_list,label='lagrange with uniform points')
cs1,=plt.semilogy(N_list,cs1_list,label='cubic spines with uniform points')
lc1,=plt.semilogy(N_list,lc1_list,label='lagrange with chebyshev points')
plt.xlabel('n')
plt.ylabel('norm of error')
plt.legend(handles=[lu1,cs1,lc1],bbox_to_anchor=(0.6, 0.6))
plt.title('norm of error vs n for function1')
#Function2
plt.figure()
lu2,=plt.semilogy(N_list,lu2_list,label='lagrange with uniform points')
cs2,=plt.semilogy(N_list,cs2_list,label='cubic spines with uniform points')
lc2,=plt.semilogy(N_list,lc2_list,label='lagrange with chebyshev points')
plt.xlabel('n')
plt.ylabel('norm of error')
plt.legend(handles=[lu2,cs2,lc2],bbox_to_anchor=(0.6, 0.6))
plt.title('norm of error vs n for function2')      
print('Cubic Spines is the most accurate, Lagrange with chebyshev points is the second accurate, Lagrange with uniform points is the least accurate')
import numpy as np
import matplotlib.pyplot as plt

def exact_sol(x, t):
    return np.exp(-400*(x-t)*(x-t))

def FDM(u, dx, f1, f2, f3, A):
    f = A @ u / (-2*dx) 
    f = np.insert(f,[0,len(f)],[0,0])
    return f, f1, f2

def AB3(u, dt, f1, f2, f3):
    return u + dt * (23/12*f1 -16/12*f2 + 5/12*f3)

#part 1
delta_x = [0.02, 0.01, 0.005, 0.002]
err = []

for i in range(4):
    
    plt.figure()
    dx = delta_x[i]
    n = int(3/dx)+1
    x_list = np.linspace(-1, 2, n)
    A = np.zeros((n-2,n))
    for i in range(n-2):
        A[i,i]=-1
        A[i,i+2]=1
    dt = 0.5 * dx
    
    sol1 = exact_sol(x_list, 1) 
    u0 = exact_sol(x_list, 0)
    u1 = exact_sol(x_list, -dt)
    u2 = exact_sol(x_list, -2*dt)
    u = u0
    
    f1, d1, d2 = FDM(u0, dx, 0, 0, 0, A)
    f2, d1, d2 = FDM(u1, dx, 0, 0, 0, A)
    f3, d1, d2 = FDM(u2, dx, 0, 0, 0, A)
    
    for i in range(n*2):      
        u = AB3(u, dt, f1, f2, f3)
        f1, f2, f3 = FDM(u, dx, f1, f2, f3, A)
        
    err.append(np.max(np.abs(np.abs(u) - sol1)))
    plt.title("dx = " + str(dx))
    plt.xlabel("x")
    plt.ylabel("u")
    plt.plot(x_list, u0, label = "t = 0")
    plt.plot(x_list, np.abs(u), label = "t = 1")
    plt.legend()

plt.figure()
plt.title("Error vs Delta x")
plt.xlabel("delta x")
plt.ylabel("error")
plt.loglog(delta_x, err,label="err")
#plt.loglog(delta_x,delta_x,label="O(n)")
plt.loglog(delta_x,np.power(delta_x,2),label="O(n^2)")
#plt.loglog(delta_x,np.power(delta_x,3),label="O(n^3)")
plt.legend()

#part 2
plt.figure()
dx = 0.002
n = int(3/dx)+1
x_list = np.linspace(-1, 2, n)
A = np.zeros((n-2,n))
for i in range(n-2):
    A[i,i]=-1
    A[i,i+2]=1
dt = 0.7 * dx

sol1 = exact_sol(x_list, 1)
u0 = exact_sol(x_list, 0)
u1 = exact_sol(x_list, -dt)
u2 = exact_sol(x_list, -2*dt)
u = u0

f1, d1, d2 = FDM(u0, dx, 0, 0, 0, A)
f2, d1, d2 = FDM(u1, dx, 0, 0, 0, A)
f3, d1, d2 = FDM(u2, dx, 0, 0, 0, A)

for i in range(715):      
    u = AB3(u, dt, f1, f2, f3)
    f1, f2, f3 = FDM(u, dx, f1, f2, f3, A)

plt.title("CFL=0.7")
plt.xlabel("x")
plt.ylabel("u")
plt.plot(x_list, u0, label = "initial condition")
plt.plot(x_list, np.abs(u), label = "solution at t = 1")
plt.legend()

 
plt.figure()
dx = 0.002
n = int(3/dx)+1
x_list = np.linspace(-1, 2, n)
A = np.zeros((n-2,n))
for i in range(n-2):
    A[i,i]=-1
    A[i,i+2]=1
dt = 0.75 * dx

sol1 = exact_sol(x_list, 1)
u0 = exact_sol(x_list, 0)
u1 = exact_sol(x_list, -dt)
u2 = exact_sol(x_list, -2*dt)
u = u0

f1, d1, d2 = FDM(u0, dx, 0, 0, 0, A)
f2, d1, d2 = FDM(u1, dx, 0, 0, 0, A)
f3, d1, d2 = FDM(u2, dx, 0, 0, 0, A)

for i in range(667):      
    u = AB3(u, dt, f1, f2, f3)
    f1, f2, f3 = FDM(u, dx, f1, f2, f3, A)

plt.title("CFL=0.75")
plt.xlabel("x")
plt.ylabel("u")
plt.plot(x_list, u0, label = "initial condition")
plt.plot(x_list, np.abs(u), label = "solution at t = 1")
plt.legend()

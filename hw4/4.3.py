import numpy as np
import numpy.linalg as la
import scipy.optimize as opt

######## EXAMPLE FOR USING MINIMIZE_SCALAR ##############
## Define function
#def f(x,y,s):
#    return s*(y+x)
## Call routine - min now contains the minimum x for the function
#min = opt.minimize_scalar(f,args=(y,s)).x

#########################################################

def rosenbrock(alpha,x):
    x = x-alpha*gradient(x)
    x1 = x[0]
    x2 = x[1]
    return 100*(x2 - x1**2)**2 + (1-x1)**2

# FINISH THIS
def gradient(x):
    # Returns gradient of rosenbrock function at x as numpy array
    x1 = x[0]
    x2 = x[1]
    grad=np.array([-400*x1*(x2-x1**2)-2*(1-x1),200*(x2-x1**2)])
    return grad

# FINISH THIS
def hessian(x):
    # Returns hessian of rosenbrock function at x as numpy array
    x1 = x[0]
    x2 = x[1]
    hess = np.array([[-400*(x2-3*x1**2)+2,-400*x1],[-400*x1,200]])
    return hess

# INSERT NEWTON FUNCTION DEFINITION
def nm(x):
    for i in range(1,11):
        s=la.solve(hessian(x),-gradient(x))
        x=x+s
    return x
# INSERT STEEPEST DESCENT FUNCTION DEFINITION
def sd(x):
    for i in range(1,11):
        alpha=opt.minimize_scalar(rosenbrock,args=(x)).x
        x=x-alpha*gradient(x)
    return x

# DEFINE STARTING POINTS AND RETURN SOLUTIONS
start1 = np.array([-1.,1.])
start2 = np.array([0.,1.])
start3 = np.array([2.,1.])

sd1 = sd(start1)
sd2 = sd(start2)
sd3 = sd(start3)

nm1 = nm(start1)
nm2 = nm(start2)
nm3 = nm(start3)
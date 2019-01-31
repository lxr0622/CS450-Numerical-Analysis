import numpy as np
from math import factorial

#relative error
negative=np.ones(5)

#approximate the exponential function    
def approximation(x,n):
    #approximated exponential function
    exp=0.0 
    #summate to nth term
    for i in range(n):
        exp+=(x**i)/factorial(i)
    return exp

#counter
j=0
n=1

#input
for x in [1,5,10,15,20]:
    #when the increment of summation is smaller than machine precision, break the loop and summation stop
    while np.abs((x**n)/factorial(n))>=np.finfo(float).eps:
        n=n+1
    #calculate relative error and add it to the list
    #for approxiamted value: e^-x=1/e^x
    negative[j]=(np.abs(1/approximation(x,n)-np.exp(-x))/np.exp(-x))
    j+=1

print(negative)

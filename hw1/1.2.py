import numpy as np
import matplotlib.pyplot as plt

## Change the following functions to reflect the
## formula on question 1.

def fd_approx(x, h):
    return (np.cos(x-2*h) - 2*np.cos(x-h)+np.cos(x))/(h*h)

def truncation_error(x, h):
    return np.abs(h*np.sin(x))

def roundoff_error(x, h):
    return np.abs(4*np.finfo(float).eps*np.cos(x)/(h*h))

## Main code
x         = 1.0;
true_val  = -np.cos(x)

h_list       = []
total_e_list = []
trunc_e_list = []
round_e_list =  []

for k in range(100):
    h = 1.5**(-k)
    h_list.append(h)

    approx_val  = fd_approx(x, h)
    trunc_error = truncation_error(x, h)
    round_error = roundoff_error(x, h)

    total_e_list.append(np.abs(true_val - approx_val))
    trunc_e_list.append(trunc_error)
    round_e_list.append(round_error)

## Plot the data
plt.loglog(h_list, trunc_e_list, '--', label='truncation error')
plt.loglog(h_list, round_e_list, '--', label='roundoff error')
plt.loglog(h_list, total_e_list, '-o', label='total error')

plt.xlabel('step size')
plt.ylabel('error')
plt.legend(frameon=False)
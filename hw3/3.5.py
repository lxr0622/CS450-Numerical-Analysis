import numpy as np



def lanczos(A, x0, iterations):
    #intial condition:
    q=np.zeros((np.shape(A)[0],iterations+2))
    q[:,1]=x0/np.linalg.norm(x0)
    T=np.zeros((iterations,iterations))
    beta_0=0
    #iterations
    for k in range(iterations):
        u_k=A@q[:,k+1]
        alpha_k=np.transpose(q[:,k+1])@u_k
        u_k=u_k-beta_0*q[:,k]-alpha_k*q[:,k+1]
        beta_k=np.linalg.norm(u_k)
        if beta_k==0:
            break
        q[:,k+2]=u_k/beta_k
        beta_0=beta_k
        #buld T
        T[k][k]=alpha_k
        if k==iterations-1:
            break
        T[k][k+1]=beta_k
        T[k+1][k]=beta_k
    #build Q
    Q=np.zeros((np.shape(A)[0],iterations))
    for i in range(iterations):
        Q[:,i]=q[:,i+1]

    return Q, T


import numpy as np
import matplotlib.pyplot as plt

#list for hilbert matrix
hilbert=[]
for n in range(2,13):
    h=np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            h[i][j]=1/(i+j+1)
    hilbert.append(h)

#classical Gram Schmidt funciton
def CGS(A):
    a=A.copy()
    n=len(a)
    #create q, to be orthogonal matrix
    q=a.copy()
    for k in range(n):
        for j in range(k):
            r_jk=np.dot(np.transpose(q[:,j]),a[:,k])
            q[:,k]=q[:,k]-r_jk*q[:,j]
        r_kk=np.linalg.norm(q[:,k])
        #if it has a linear relationship, break the loop
        if r_kk==0:
            break
        q[:,k]=q[:,k]/r_kk
    return q

#Modified Gram Schmidt function
def MGS(A):
    a=A.copy()
    n=len(a)
    #create q, to be orthogonal matrix
    q=a.copy()
    for k in range(n):
        r_kk=np.linalg.norm(a[:,k])
        #if it has a linear relationship, break the loop
        if r_kk==0:
            break
        q[:,k]=a[:,k]/r_kk
        for j in range(k+1,n):
            r_kj=np.dot(np.transpose(q[:,k]),a[:,j])
            a[:,j]=a[:,j]-r_kj*q[:,k]
    return q

#Householder
def HH(A):
    n=len(A)
    a=A.copy()
    I=np.identity(n)
    q=np.identity(n)
    for k in range(n):
        alpha=-np.sign(a[k][k])*np.linalg.norm(a[:,k])
        e = np.zeros(n)
        e[k] = 1
        v=np.transpose(a[:,k]-alpha*np.array([e]))
        H=I-2*np.dot(v,np.transpose(v))/np.dot(np.transpose(v),v)
        q=np.dot(q,np.transpose(H))
    return q
        
     
#digits of accuracy
n_list=np.arange(2,13).tolist()
accuracy_CGS=[] 
accuracy_CGS2=[]
accuracy_MGS=[]
accuracy_HH=[]       
for i in range(11):
    
    #identity matrix
    I=np.identity(n_list[i])
    
    #CGS method
    Q=CGS(hilbert[i])
    accuracy_CGS.append(-np.log10(np.linalg.norm(I-np.dot(np.transpose(Q),Q))))
    
    #CGS method by twice
    Q=CGS(hilbert[i])
    Q2=CGS(Q)
    accuracy_CGS2.append(-np.log10(np.linalg.norm(I-np.dot(np.transpose(Q2),Q2))))
    
    #MGS method
    Q=MGS(hilbert[i])
    accuracy_MGS.append(-np.log10(np.linalg.norm(I-np.dot(np.transpose(Q),Q))))
    
    #Householder method
    Q=HH(hilbert[i])
    accuracy_HH.append(-np.log10(np.linalg.norm(I-np.dot(np.transpose(Q),Q))))


#plot
CGS,=plt.plot(n_list,accuracy_CGS,label='Classic Gram-Schmidt')
CGS2,=plt.plot(n_list,accuracy_CGS2,label='Classic Gram-Schmidt by twice')
MGS,=plt.plot(n_list,accuracy_MGS,label='Modified Gram-Schmidt')
HH,=plt.plot(n_list,accuracy_HH,label='Householder')
plt.xlabel('n')
plt.ylabel('Accuracy')
plt.legend(handles=[CGS,CGS2,MGS,HH])
plt.title('Accuracy of Different QR Implementations')
    
            
            
            
            
        
        
    


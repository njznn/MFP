import numpy as np
import matplotlib.pyplot as plt
import sys

from numpy import linalg as LA
epsilon = 1e-5
#A je mxn matrika
#Q je mxm ortogonalna
#R=rezultat, koncna A je 'zgornje' trikotna

def qr(M):
    A = np.copy(M)
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A

def make_householder(a):
    #rescaling to v and sign for numerical stability
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.outer(v,v)
    return H

def qr_givens(M):
    A = np.copy(M)
    m, n = A.shape
    Q = np.eye(m)
    for j in range(n - (m == n)):
        for i in range(j+1,m):
            r=np.hypot(A[j,j],A[i,j])
            c=A[j,j]/r
            s=A[i,j]/r
            givensRot = np.array([[c, s],[-s,  c]])
            A[[j,i],j:] = np.dot(givensRot, A[[j,i],j:])
            Q[[j,i],:] = np.dot(givensRot, Q[[j,i],:])
    return Q.T, A

def trid_householder(M):
    A = np.copy(M)
    m, n = A.shape
    if ( m != n):
        print("need quadratic symmetric matrix")
        sys.exit(1)
    Q = np.eye(m)
    for i in range(m - 2):
        H = np.eye(m)
        H[i+1:, i+1:] = make_householder(A[i+1:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
        A = np.dot(A,H)
    return Q, A


def QR_iteracija(F,Q,R, n_max):
    A = np.copy(Q)
    B = np.copy(R)
    C = B@A
    QQ = np.eye(len(A))
    i = 0
    while  not (np.all(abs(np.sort(np.diag(C))-LA.eigh(F)[0]))< epsilon):
        A,B = qr(C)
        C = B@A
        QQ = QQ@A
        i +=1
        if i==n_max:
            break
        else:
            None
    return(np.sort(np.diag(C)), Q, i)

def QR_iteracija_giv(F,Q,R, n_max):
    A = np.copy(Q)
    B = np.copy(R)
    C = B@A
    QQ = np.eye(len(A))
    i = 0
    while  not (np.all(abs(np.sort(np.diag(C))-LA.eigh(F)[0]))< epsilon):
        A,B = qr_givens(C)
        C = B@A
        QQ = QQ@A
        i +=1
        if i==n_max:
            break
        else:
            None
    return(np.sort(np.diag(C)), QQ,i)

def qlnr(d,e,z,tol = 1.0e-9):
    #d - diagonal values
    #e - off-tridiag values
    #z - orthogonal matrix to process further
    n=len(d)
    e=np.roll(e,-1) #reorder
    itmax=1000
    for l in range(n):
        for iter in range(itmax):
            m=n-1
            for mm in range(l,n-1):
                dd=abs(d[mm])+abs(d[mm+1])
                if abs(e[mm])+dd == dd:
                    m=mm
                    break
                if abs(e[mm]) < tol:
                    m=mm
                    break
            if iter==itmax-1:
                print ("too many iterations",iter)
                break
            if m!=l:
                g=(d[l+1]-d[l])/(2.*e[l])
                r=np.sqrt(g*g+1.)
                g=d[m]-d[l]+e[l]/(g+np.sign(g)*r)
                s=1.
                c=1.
                p=0.
                for i in range(m-1,l-1,-1):
                    f=s*e[i]
                    b=c*e[i]
                    if abs(f) > abs(g):
                        c=g/f
                        r=np.sqrt(c*c+1.)
                        e[i+1]=f*r
                        s=1./r
                        c *= s
                    else:
                        s=f/g
                        r=np.sqrt(s*s+1.)
                        e[i+1]=g*r
                        c=1./r
                        s *= c
                    g=d[i+1]-p
                    r=(d[i]-g)*s+2.*c*b
                    p=s*r
                    d[i+1]=g+p
                    g=c*r-b
                    for k in range(n):
                        f=z[k,i+1]
                        z[k,i+1]=s*z[k,i]+c*f
                        z[k,i]=c*z[k,i]-s*f
                d[l] -= p
                e[l]=g
                e[m]=0.
            else:
                break
    return(d,z)

def trid_qlnr(H):
    A = np.copy(H)
    q,r=trid_householder(A)
    n=r.shape[0]
    d=np.zeros(n)
    e=np.zeros(n)
    for i in range(n):
        d[i]=r[i,i]
        for i in range(n-1):
            e[i+1]=r[i+1,i]
    lastne, vektorji = qlnr(d,e,q)
    return(lastne, vektorji)

#q,r = trid_householder(H)
#n=r.shape[0]
#d=np.zeros(n)
#e=np.zeros(n)
#for i in range(n):
#    d[i]=r[i,i]
#for i in range(n-1):
#    e[i+1]=r[i+1,i]


##podobno kot smo pri numeričnih metodah:###
def jakobi_iteracija(B,epsilon = 1e-6):
    A = np.copy(B)
    def najvecji_element(A):
        n = len(A)
        max = 0.0
        for i in range(n-1):
            for j in range(i+1,n):
                if abs(A[i,j]) >= max:
                    max = abs(A[i,j])
                    k = i
                    l = j
        return(max,k,l)

    def rotacija_elementa(A,p,k,l):
        n = len(A)
        razlika = A[l,l] - A[k,k]
        if abs(A[k,l]) < abs(razlika)*1.0e-30:
            t = A[k,l]/razlika
        else:
            phi = razlika/(2.0*A[k,l])
            t = 1.0/(abs(phi) + np.sqrt(phi**2 + 1.0))
            if phi < 0.0:
                t = -t
        c = 1.0/np.sqrt(t**2 + 1.0)
        s = t*c
        tau = s/(1.0 + c)
        temp = A[k,l]
        A[k,l] = 0.0
        A[k,k] = A[k,k] - t*temp
        A[l,l] = A[l,l] + t*temp
        for i in range(k):
            temp = A[i,k]
            A[i,k] = temp - s*(A[i,l] + tau*temp)
            A[i,l] = A[i,l] + s*(temp - tau*A[i,l])
        for i in range(k+1,l):
            temp = A[k,i]
            A[k,i] = temp - s*(A[i,l] + tau*A[k,i])
            A[i,l] = A[i,l] + s*(temp - tau*A[i,l])
        for i in range(l+1,n):
            temp = A[k,i]
            A[k,i] = temp - s*(A[l,i] + tau*temp)
            A[l,i] = A[l,i] + s*(temp - tau*A[l,i])
        for i in range(n):
            temp = p[i,k]
            p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
            p[i,l] = p[i,l] + s*(temp - tau*p[i,l])

    A=np.copy(B)
    n = len(B)
    najvec_rotacij = 5*(n**2)
    P = np.identity(n)
    for i in range(najvec_rotacij):
        max,k,l = najvecji_element(A)
        if max < epsilon:
            return(np.diagonal(A),P)
        rotacija_elementa(A,P,k,l)

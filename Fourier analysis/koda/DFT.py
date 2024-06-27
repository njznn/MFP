import numpy as np



def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1)) # in essence row->column transformation, k*n is then the dot product..
    #print("Vector",k)
    #print("Matrix",k*n)
    M = np.exp(-2j * np.pi * k * n / N)
    #print("Matrix",M)
    return np.dot(M, x)

def DFT_simplest(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    F=[]
    for k in range(N):
            fk=0.
            for n in range(N):
                Mkn=np.exp(-2j * np.pi * k * n / N)
                fk += Mkn*x[n]
            F.append(fk)
            #print("k,F[k]",k,fk)
    return np.asarray(F)

def DFT_faster(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.T # in essence row->column transformation, k*n is then the dot product..
    #print("Vector",k)
    #print("Matrix",k*n)
    vektor = np.array([])
    for i in range(N):
        M_i = np.exp(-2j * np.pi * k[i] * n/N)
        vektor = np.append(vektor, np.dot(M_i, x))
    #print("Matrix",M)
        #print(np.dot(M_i, x))
    return(vektor)


def DFT_slow_inv(w):
    #w = np.asarray(w, dtype=float)
    N = w.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1)) # in essence row->column transformation, k*n is then the dot product..
    #print("Vector",k)
    #print("Matrix",k*n)
    M = np.exp(2j * np.pi * k * n / N)
    #print("Matrix",M)
    return np.dot(M, w)

def DFT_simplest_inv(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    #x = np.asarray(x, dtype=float)
    N = x.shape[0]
    F=[]
    for k in range(N):
            fk=0.
            for n in range(N):
                Mkn=np.exp(2j * np.pi * k * n / N)
                fk += Mkn*x[n]
            F.append(fk)
            #print("k,F[k]",k,fk)
    return np.asarray(F)

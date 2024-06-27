from metode_qr import *
from hamiltonke import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy import linalg as LA
from scipy.linalg import schur
from timeit import default_timer as timer
import scipy.linalg as lin
import scipy.linalg.lapack as la



t1 = timer()
H = hamiltonka(1, 0.5, 10000)
t2 = timer()
t3 = timer()
H1 = hamiltonka(2, 0.5, 10000)
t4 = timer()
t5 = timer()
H3 = hamiltonka(3, 0.5, 10000)
t6 = timer()
print(t2-t1, t4-t3, t6-t5)
#######casovna_zahtevnost############
casi_eigh = np.array([])
casi_svd = np.array([])
casi_qr = np.array([])
casi_qr_giv = np.array([])
casi_jacobi = np.array([])
casi_trid = np.array([])
casi_trid_qnlr = np.array([])
#N = np.array([i for i in range(2, 40)])
#lambda = 0.2, 1
#for i in range(2, 40):
#    H = hamiltonka(1,0.2,i)
    #t_eigh1 = timer()
    #lambda_b, lastni = LA.eigh(H)
    #print(lambda_b)
    #t_eigh_k = timer()
    #casi_eigh = np.append(casi_eigh, t_eigh_k-t_eigh)
    #t_svd = timer()
    #U,s,VT = LA.svd(H)
    #t_svd_k = timer()
    #casi_svd = np.append(casi_svd, t_svd_k-t_svd)
    #t_qr = timer()
    #q,r = qr(H)
    #a = QR_iteracija(H,q,r,200)
    #print(a[0])
    #t_qr_k = timer()
    #casi_qr = np.append(casi_qr, t_qr_k-t_qr)
    #t_qr_giv = timer()
    #q1,r1 = qr_givens(H)
    #QR_iteracija_giv(H,q1,r1,200)
    #t_qr_giv_k = timer()
    #casi_qr_giv = np.append(casi_qr_giv, t_qr_giv_k-t_qr_giv)
    #t_jacobi = timer()
    #jakobi_iteracija(H, 1e-10)
    #t_jacobi_k = timer()
    #casi_jacobi = np.append(casi_jacobi, t_jacobi_k-t_jacobi)
    #t_trid_qlnr = timer()
    #trid_qlnr(H)
    #t_trid_qnlr_t = timer()
    #casi_trid_qnlr = np.append(casi_trid_qnlr, t_trid_qnlr_t-t_trid_qlnr)

#plt.plot(N, casi_eigh, label ='numpy-eigh')
#plt.plot(N, casi_svd, label='numpy-svd')
#plt.plot(N, casi_qr, label='housholder + qr')
#plt.plot(N, casi_qr_giv, label='givnes + qr')
#plt.plot(N, casi_jacobi, label = 'jacobi')
#plt.plot(N, casi_trid_qnlr, label = 'tridiag_hous + qnlr')
#plt.xlabel('N')
#plt.ylabel('čas[s]')
#plt.title('b)')
#plt.legend()
#plt.show()
#plt.clf()

################natancnost_lastnih_vrednosti#############
#plt.plot(N, LA.norm(lambda_b - s), label = '$\Delta(svd)')
N= 10
H = hamiltonka(1,0.2,N)
lambda_b, lastni = LA.eigh(H)
U,s,VT = LA.svd(H)
q,r = qr(H)
a = QR_iteracija(H,q,r,200)
q1,r1 = qr_givens(H)
b = QR_iteracija_giv(H,q1,r1,200)
c =jakobi_iteracija(H, 1e-10)
d = trid_qlnr(H)
#plt.plot([i for i in range(1, N+1)], abs(lambda_b -a[0]) , label = r'$\Delta Householder$')
plt.plot([i for i in range(1, N+1)], abs(lambda_b -np.sort(s)), label = r'$\Delta SVD$' )
#plt.plot([i for i in range(1, N+1)], abs(lambda_b -np.sort(b[0])), label =r'$\Delta Givens$' )
plt.plot([i for i in range(1, N+1)], abs(lambda_b -np.sort(c[0])), label =r'$\Delta Jacobi$' )
plt.plot([i for i in range(1, N+1)], abs(lambda_b -np.sort(d[0])), 'y', label =r'$\Delta Trid-QRNL$' )
plt.xlabel('N')
plt.ylabel(r'$|\lambda_{i-eigh} - \lambda_{i}|$')
plt.title('b)')
plt.tight_layout()
plt.legend()
#plt.show()

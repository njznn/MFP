import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math
import scipy.special as spec
import sys
from mpmath import mp
from decimal import*

def arijeve_poz_A(x, n, epsilon):
    L_ = 1.
    L = np.array([])
    L = np.append(L, L_)
    dodatek = 1.
    Q = (2./3.)*abs(x)**(3./2.)
    for i in range(1, n+1):
        dodatek = ((dodatek/(-Q))*((3.*(i-1)+5./2.)*(3.*(i-1)+3./2.)*(3.*(i-1)+1./2.))/(54.*(i)*(i+1./2.)))
        stara = L_
        L_ += dodatek
        L = np.append(L, L[-1] + dodatek)
        #print('star:{}, nov:{}, dodatek:{}, iteracija:{}'.format(stara, L_, dodatek, i))
        if abs(dodatek)<epsilon:
        #if i==n:
        #if (abs(dodatek/L[-1])) < epsilon:
            A = ((mp.exp(-Q))*L_)/(2*mp.sqrt(mp.pi)*x**(1/4))
            #print('A:{}, tocno:{}, razlika:{}'.format(A,spec.airy(x)[0], abs(A-spec.airy(x)[0])))
            return(A,spec.airy(x)[0])
            break

def arijeve_poz_B(x,n,epsilon):
    L_plus = 1.
    dodatek_ = 1.
    Q = (2/3)*abs(x)**(3/2)
    for i in range(1,n+1):
        dodatek_ = ((dodatek_/(Q))*((3*(i-1)+5/2)*(3*(i-1)+3/2)*(3*(i-1)+1/2))/(54*(i)*(i+1/2)))
        stara_L = L_plus
        L_plus += dodatek_
        #print('star:{}, nov:{}, dodatek:{}, iteracija:{}'.format(stara_L, L_plus, dodatek_, i))
        #if abs(dodatek_/L_plus)<epsilon:
        if abs(dodatek_)<epsilon:
            B = (mp.exp(Q)*L_plus)/(mp.sqrt(mp.pi)*x**(1/4))
            #print('B:{}, tocno:{},rel_razlika:{}'.format(B,spec.airy(x)[2], abs(B-spec.airy(x)[2])))
            return(B, spec.airy(x)[2])
            break

def arijeve_nic_A(x,n,epsilon):
    a = 0.355028053887817239
    b = 0.258819403792806798
    f = 1.
    g = x
    vrednost = 1.
    vrednost_g = x
    for i in range(1, n+1):
        vrednost = (1/3+i-1)*vrednost*(3*x**3/((3*i)*(3*i-1)*(3*i-2)))
        f_stara = f
        f += vrednost
        vrednost_g = (2/3 + i-1)*vrednost_g*(3*x**3/((3*i+1)*(3*i)*(3*i-1)))
        g_stara = g
        g += vrednost_g
        A = a*f - b*g
        #print('star_f:{}, nov_f:{}, dodatek_f:{}, iteracija:{}'.format(f_stara, f, vrednost, i))
        #print('star_g:{}, nov_g:{}, dodatek_g:{}, iteracija:{}'.format(g_stara,g , vrednost_g, i))
        if  (abs(vrednost)+abs(vrednost_g)) < epsilon:
            #print("Koncam z rezultatom {:.20f}, tocno {:.20f}, razlika {:.20f}".format(A, spec.airy(x)[0], abs(A-spec.airy(x)[0])))
            return(A, spec.airy(x)[0])
            break

def arijeve_nic_B(x,n,epsilon):
    a = 0.355028053887817239
    b = 0.258819403792806798
    f = 1.
    g = x
    vrednost = 1.
    vrednost_g = x
    for i in range(1, n+1):
        #print ("Iteracija {}".format(i))
        vrednost = (1/3+i-1)*vrednost*(3*x**3/((3*i)*(3*i-1)*(3*i-2)))
        f_stara = f
        f += vrednost
        vrednost_g = (2/3 + i-1)*vrednost_g*(3*x**3/((3*i+1)*(3*i)*(3*i-1)))
        g_stara = g
        g += vrednost_g
        B = np.sqrt(3)*(a*f + b*g)
        #print('star_f:{}, nov_f:{}, dodatek_f:{}, iteracija:{}'.format(f_stara, f, vrednost, i))
        #print('star_g:{}, nov_g:{}, dodatek_g:{}, iteracija:{}'.format(g_stara,g , vrednost_g, i))
        if  abs(vrednost+vrednost_g) < epsilon:
            #print("Koncam z rezultatom {:.20f}, tocno {:.20f}, razlika {:.20f}".format(B, spec.airy(x)[2], abs(B-spec.airy(x)[2])))
            return(B, spec.airy(x)[2])
            break
def P_vrsta(x, n, epsilon):
    P = 1.
    dodatek_P = 1.
    Q = (2./3.)*abs(x)**(3./2.)
    for i in range(1, n+1):
        dodatek_P = ((-1)*(dodatek_P/(Q**2.))*((6.*(i-1)+11./2.)*(6.*(i-1)+9./2.)*(6.*(i-1)+7./2.)*(6.*(i-1)+5./2.)*(6.*(i-1)+3./2.)*(6.*(i-1)+1./2.))/((54.**2.)*(2.*(i-1)+3./2.)*(2.*(i-1)+1./2.)*(2.*i-1)*(2.*i)))
        star_P = P
        P += dodatek_P
        #print('star_P:{}, nov_P:{}, dodatek:{}, iteracija:{}'.format(star_P, P, dodatek_P, i))
        #if i==n:
        if abs(dodatek_P) < epsilon:
            print(P)
            return(P)

def Q_vrsta(x,n,epsilon):
    Q = (2/3)*abs(x)**(3/2)
    Q_0 = 15./(Q*216.)
    dodatek_Q_0 = 15./(Q*216.)
    for i in range(1,n+1):
        dodatek_Q_0 = ((-1)*(dodatek_Q_0/(Q**2))*((6*(i-1)+8+1/2)*(6*(i-1)+7+1/2)*(6*(i-1)+6+1/2)*(6*(i-1)+5+1/2)*(6*(i-1)+4+1/2)*(6*(i-1)+3+1/2))/((54**2)*(2*i+1)*(2*i)*(2*(i-1)+2+1/2)*(2*(i-1)+1+1/2)))
        star_Q = Q_0
        Q_0 += dodatek_Q_0
        #print('star_Q:{}, nov_Q:{}, dodatek:{}, iteracija:{}'.format(star_Q, Q_0, dodatek_Q_0, i))
        if abs(dodatek_Q_0) < epsilon:
        #if i== n:
            print(Q_0)
            return(Q_0)


def arijeve_min_A(x, n, epsilon):
    Q = (2/3)*abs(x)**(3/2)
    Q_0 = Q_vrsta(x,n,epsilon)
    P = P_vrsta(x,n,epsilon)
    A = ((1/(np.sqrt(np.pi)*(-x)**(1./4.)))*(np.sin(Q-np.pi/4)*Q_0 + np.cos(Q-np.pi/4)*P))
    #print('A:{}, tocno:{}, mpmath:{}'.format(A,spec.airy(x)[0], mp.airyai(x)))
    return(A, spec.airy(x)[0])


def arijeve_min_B(x, n, epsilon):
    Q = (2/3)*abs(x)**(3/2)
    Q_0 = Q_vrsta(x,n,epsilon)
    P = P_vrsta(x,n,epsilon)
    B = ((1/(mp.sqrt(mp.pi)*(-x)**(1./4.)))*(-mp.sin(Q-mp.pi/4)*P + mp.cos(Q-mp.pi/4)*Q_0))
    #B = ((1/(mp.sqrt(mp.pi)*(-x)**(1./4.)))*(-mp.sin(Q-mp.pi/4)*P + mp.cos(Q-mp.pi/4)*Q_0))
    #print('B:{}, tocno:{},razlika:{}'.format(B,spec.airy(x)[2], abs(B-spec.airy(x)[2])))
    return(B, spec.airy(x)[2])
#arijeve_min_B(-10, 100, 1e-10)

#n = max stevilo korakov:

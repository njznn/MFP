import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math
import scipy.special as spec
import sys
from vrednost_v_x import *

epsilon = 1e-10
koraki = 500
def abs_rel(a, b, st_tock, f, f_prava):
    a = np.linspace(a, b, st_tock)
    ab = np.array([])
    rel = np.array([])
    for i in a:
        ab = np.append(ab, abs(f(i)-f_prava(i)))
        rel = np.append(rel, abs(f(i)-f_prava(i))/f_prava(i))
    return(a, ab, rel)

f = lambda x: arijeve_min_A(x, koraki, epsilon)[0]
b = lambda x: arijeve_min_B(x, koraki, epsilon)[0]
g = lambda x: spec.airy(x)[0]
h = lambda x: spec.airy(x)[2]
f_0 = lambda x: arijeve_nic_A(x, koraki, epsilon)[0]
b_0 = lambda x: arijeve_nic_B(x, koraki, epsilon)[0]
f_2 = lambda x: arijeve_poz_A(x, koraki, epsilon)[0]
b_2 = lambda x: arijeve_poz_B(x, koraki, epsilon)[0]
############################################################################
tocke = abs_rel(-30, -7, 100, f, g)
tocke2 = abs_rel(-30, -7, 100, b, h)
plt.yscale('log')
plt.plot(tocke[0], tocke[2], 'ro',markersize='3', label = '$log(|Ai(x)-Ai(x)_{priblizek}/Ai(x)|)$')
plt.plot(tocke2[0], tocke2[2], 'bo', markersize='3', label = '$log(|Bi(x)-Bi(x)_{priblizek}/Bi(x)|)$')
plt.ylabel('$log(|y-y_{priblizek}/y|)$')
plt.xlabel('x')
plt.legend()
#plt.show()
plt.clf()
plt.yscale('log')
tocke_0 = abs_rel(-10,10,200, f_0, g)
tocke_01 = abs_rel(-10, 10, 200, b_0, h)
plt.plot(tocke_0[0], tocke_0[2], 'ro',markersize='3', label = '$log(|Ai(x)-Ai(x)_{priblizek}|/Ai(x))$')
plt.plot(tocke_01[0], tocke_01[2], 'bo', markersize='3', label = '$log(|Bi(x)-Bi(x)_{priblizek}|Bi(x))$')
plt.ylabel('$log(|y-y_{priblizek}|/y)$')
plt.xlabel('x')
plt.legend(loc = 'upper center', fontsize = 'small')
#plt.show()
plt.clf()
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.set_yscale('log')
ax2.set_yscale('log')
tocke_30 = abs_rel(6, 50, 1000, f_2, g)
tocke_31 = abs_rel(6, 50,1000, b_2, h)
ax1.plot(tocke_30[0], tocke_30[1], 'ro',markersize='3', label = '$log(|Ai(x)-Ai(x)_{priblizek}/Ai(x)|)$')
ax2.plot(tocke_31[0], tocke_31[1], 'bo', markersize='3', label = '$log(|Bi(x)-Bi(x)_{priblizek}/Bi(x)|)$')
ax1.set_ylabel('$log(|y-y_{priblizek}/y|)$')
ax1.set_xlabel('x')
ax2.set_xlabel('x')
ax2.legend(loc = 'upper left', fontsize = 'small')
ax1.legend(loc = 'upper left', fontsize = 'small')
plt.show()
plt.clf()

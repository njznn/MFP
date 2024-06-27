import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math
import scipy.special as spec
import sys
from vrednost_v_x import *

epsilon = 1e-10
koraki = 100
j1 = []
j2 = []
j3 = []
j1b = []
j2b = []
j3b = []

A_min = []
A_nic = []
A_poz = []
B_min = []
B_nic = []
B_poz = []
j = np.linspace(-30, 20, 500)
for x in j:
    if x <= (-8):
        j1.append(x)
        A_min.append(arijeve_min_A(x, koraki, epsilon)[0])
    elif -8 < x < 6:
        j2.append(x)
        A_nic.append(arijeve_nic_A(x, koraki, epsilon)[0])
    elif x >= 6:
        j3.append(x)
        A_poz.append(arijeve_poz_A(x, koraki, epsilon)[0])

for x in j:
    if x <= (-8):
        j1b.append(x)
        B_min.append(arijeve_min_B(x, koraki, epsilon)[0])
    elif -8 < x < 6:
        j2b.append(x)
        B_nic.append(arijeve_nic_B(x, koraki, epsilon)[0])
    elif  x > 6:
        j3b.append(x)
        B_poz.append(arijeve_poz_B(x, koraki, epsilon)[0])
arijeve = [spec.airy(x)[2] for x in j]
vsi_B = B_min+B_nic+B_poz





fig, axs = plt.subplots(2, 2)
axs[0,0].grid()
axs[0,0].plot(j1b, B_min ,label = 'Bi(x)-priblizek')
axs[0,0].plot(j1b, [spec.airy(x)[2] for x in j1b] ,'--', label = 'Bi(x)')
axs[0,0].set_title('Bi(x)-negativna števila',fontsize = 10)
axs[1,1].plot(j2b[40:100], B_nic[40:100] ,label = 'Bi(x)-priblizek')
axs[1,1].plot(j2b[40:100], [spec.airy(x)[2] for x in j2b[40:100]],'--', label = 'Bi(x)')
axs[1,1].set_title('Bi(x)-okrog ničle',fontsize = 10)
axs[1,0].plot(j3b, B_poz ,label = 'Bi(x)-priblizek')
axs[1,0].plot(j3b, [spec.airy(x)[2] for x in j3b],'--', label = 'Bi(x)')
axs[1,0].set_title('Bi(x)-pozitivna števila',fontsize = 10)
axs[0,1].plot(j1b+j2[:100], B_min+B_nic[:100] ,label = 'Bi(x)-priblizek')
#axs[0,1].plot(j2b, B_nic, c='#1f77b4')
axs[0,1].plot(j1b+j2b[:100], [spec.airy(x)[2] for x in j1b+j2b[:100]] ,'--',label = 'Bi(x)', color = 'C1')
axs[0,1].set_title('Bi(x)',fontsize = 10)
axs[0,0].legend(loc='lower center')
axs[0,1].legend()
axs[1,1].legend()
axs[1,0].legend()
axs[0,1].grid()
axs[1,1].grid()
axs[1,0].grid()
axs[1,0].set_xlabel('x')
axs[1,1].set_xlabel('x')
axs[0,0].set_xlabel('x')
axs[0,1].set_xlabel('x')
plt.tight_layout()

#plt.show()
plt.close()
plt.clf()
fig, axs = plt.subplots(2, 2)
axs[0,0].grid()
axs[0,0].plot(j1[:], A_min[:] ,label = 'Ai(x)-priblizek')
axs[0,0].plot(j1, [spec.airy(x)[0] for x in j1] ,'--', label = 'Ai(x)')
axs[0,0].set_title('Ai(x)-negativna števila',fontsize = 10)
axs[1,1].plot(j2, A_nic ,label = 'Ai(x)-priblizek')
axs[1,1].plot(j2, [spec.airy(x)[0] for x in j2],'--', label = 'Ai(x)')
axs[1,1].set_title('Ai(x)-okrog ničle',fontsize = 10)
axs[1,0].plot(j3[33:], A_poz[33:] ,label = 'Ai(x)-priblizek')
axs[1,0].plot(j3[33:], [spec.airy(x)[0] for x in j3][33:],'--', label = 'Ai(x)')
axs[1,0].set_title('Ai(x)-pozitivna števila',fontsize = 10)
axs[0,1].plot(j1, A_min, label= 'Ai(x)-priblizek', color = '#1f77b4')
axs[0,1].plot(j2, A_nic, color = '#1f77b4')
axs[0,1].plot(j3, A_poz, c='#1f77b4')
axs[0,1].plot(j, [spec.airy(x)[0] for x in j],'--', label='Ai(x)', color = 'C1')
axs[0,1].set_title('Ai(x)',fontsize = 10)
axs[0,0].legend()
axs[0,1].legend()
axs[1,1].legend()
axs[1,0].legend()
axs[0,1].grid()
axs[1,1].grid()
axs[1,0].grid()
axs[1,0].set_xlabel('x')
axs[1,1].set_xlabel('x')
axs[0,0].set_xlabel('x')
axs[0,1].set_xlabel('x')
plt.tight_layout()
plt.show()

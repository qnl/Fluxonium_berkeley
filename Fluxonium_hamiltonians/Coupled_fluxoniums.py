import numpy as np
from matplotlib import pyplot as plt
from qutip import *

# Define constants
e = 1.602e-19  # Fundamental charge
h = 6.626e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum

directory = 'C:\\Users\\nguyen89\Documents\Python Codes\Fluxonium simulation results'
fname = "Coupled_fluxonium_AugustusVIII_spectrum_test_N=20.txt"
path = directory + '\\' + fname

Na = 25
Nb = 25
B_coeff = 30

E_la=0.45128451566691613
E_ca=0.9605239907246711
E_ja=5.9190643872351485

E_lb=0.722809860436973
E_cb=1.0161030829798336
E_jb=5.744246275094001

J_c = 0.1
# J_c_array = np.linspace(0,1,101)
# level_num = 20
# current = np.linspace(0,1.0,51)*1e-3
# energies = np.zeros((len(current), level_num))

level_num = 20
phi_ext_array = np.linspace(0, 1, 101)
# phi_ext = 0.5
spectrum = np.zeros((len(phi_ext_array), level_num))
nem = np.zeros((len(phi_ext_array), level_num * 2))
#########################################################
for idx, phi_ext in enumerate(phi_ext_array):
    a = tensor(destroy(Na), qeye(Nb))
    phi_a = (a + a.dag()) * (8.0 * E_ca / E_la) ** (0.25) / np.sqrt(2.0)
    na_a = 1.0j * (a.dag() - a) * (E_la / (8 * E_ca)) ** (0.25) / np.sqrt(2.0)
    ope_a = 1.0j * (phi_a - 2 * np.pi * phi_ext)
    H_a = 4.0 * E_ca * na_a ** 2.0 + 0.5 * E_la * phi_a ** 2.0 - 0.5 * E_ja * (ope_a.expm() + (-ope_a).expm())

    b = tensor(qeye(Na), destroy(Nb))
    phi_b = (b + b.dag()) * (8.0 * E_cb / E_lb) ** (0.25) / np.sqrt(2.0)
    na_b = 1.0j * (b.dag() - b) * (E_lb / (8 * E_cb)) ** (0.25) / np.sqrt(2.0)
    ope_b = 1.0j * (phi_b - 2 * np.pi * phi_ext)
    H_b = 4.0 * E_cb * na_b ** 2.0 + 0.5 * E_lb * phi_b ** 2.0 - 0.5 * E_jb * (ope_b.expm() + (-ope_b).expm())

    Hc = J_c * na_a * na_b
    H = H_a + H_b + Hc
    eigenenergies, eigenstates = H.eigenstates()
    for idy in range(level_num):
        spectrum[idx, idy] = eigenenergies[idy]
    print(str(round((idx + 1) / len(phi_ext_array) * 100, 2)) + "%")

np.savetxt(path, spectrum)
#########################################################
energies = np.genfromtxt(path)

for idx in range(1,level_num):
    plt.plot(phi_ext_array, energies[:,idx]-energies[:,0])

plt.ylim([0,15])
plt.xlim([0,1])
plt.ylabel("Frequency (GHz)")
plt.xlabel('Flux (flux quantum)')
plt.show()
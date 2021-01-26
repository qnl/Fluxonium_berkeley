import numpy as np
from qutip import*
from matplotlib import pyplot as plt

def charge_ope1(N, E_L, E_C1):
    a = tensor(destroy(N), qeye(N))
    return 1.0j*(a.dag()-a)*(E_L/(8*E_C1))**(0.25)/np.sqrt(2.0)

def charge_ope2(N, E_L, E_C2):
    a = tensor(qeye(N), destroy(N))
    return 1.0j*(a.dag()-a)*(E_L/(8*E_C2))**(0.25)/np.sqrt(2.0)

def phase_ope1(N, E_L, E_C1):
    a = tensor(destroy(N), qeye(N))
    return (a + a.dag()) * (8.0 * E_C1 / E_L) ** (0.25) / np.sqrt(2.0)

def phase_ope2(N, E_L, E_C2):
    a = tensor(qeye(N), destroy(N))
    return (a + a.dag()) * (8.0 * E_C2 / E_L) ** (0.25) / np.sqrt(2.0)

def hamiltonian(N, E_L, E_C1, E_J1, E_C2, E_J2, phi_ext):
    na1 = charge_ope1(N, E_L, E_C1)
    na2 = charge_ope2(N, E_L, E_C2)
    phi1 = phase_ope1(N, E_L, E_C1)
    phi2 = phase_ope2(N, E_L, E_C2)
    H = 4*E_C1*na1**2 + 4*E_C2*na2**2 + 0.5*E_L*(phi1 - phi_ext)**2
    - E_J1*phi1.cosm()*phi2.cosm()
    return H

# params = {
#     'N': 30,
#     'E_L': 0.5,
#     'E_C1': 1,
#     'E_J1': 5,
#     'E_C2': 3.5,
#     'E_J2': 15,
#     'phi_ext': 0
# }
N = 30
E_L = 0.5
E_C1 = 1
E_C2 = 1
E_J1 = 6
E_J2 = 6

phi_ext = np.linspace(0,1,21)
energies = np.zeros((len(phi_ext),N))
me_charge1 = np.zeros_like(phi_ext)
me_charge2 = np.zeros_like(phi_ext)
me_phase1 = np.zeros_like(phi_ext)
me_phase2 = np.zeros_like(phi_ext)

for idx, phi in enumerate(phi_ext):
    H = hamiltonian(N, E_L, E_C1, E_J1, E_C2, E_J2, phi*2*np.pi)
    eigenenergies, eigenstates_ho = H.eigenstates()
    energies[idx,:] = eigenenergies[0:N]

for idx in range(10):
    plt.plot(phi_ext, energies[:,idx] - energies[:,0])

plt.show()
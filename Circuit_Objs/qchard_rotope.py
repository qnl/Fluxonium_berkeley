# Author: Long Nguyen

"""
Define and expand operators' functions to have N-level Hilbert space
"""

from qutip import*
import numpy as np
import qutip.qip.operations as qtop

def sigx(N):
    m = np.diag(np.zeros(N, dtype = complex))
    m[0,1] = 1
    m[1,0] = 1
    return Qobj(m)

def sigy(N):
    m = np.diag(np.zeros(N, dtype = complex))
    m[0, 1] = -1j
    m[1, 0] = 1j
    return Qobj(m)

def sigz(N):
    m = np.diag(np.zeros(N, dtype = complex))
    m[0,0] = 1
    m[1,1] = -1
    return Qobj(m)

def sx(phi,N):
    m = np.diag(np.zeros(N, dtype=complex))
    for i in range(2):
        for j in range(2):
            m[i, j] = qtop.rx(phi=phi).data[i,j]
    return Qobj(m)

def sy(phi,N):
    m = np.diag(np.zeros(N, dtype=complex))
    for i in range(2):
        for j in range(2):
            m[i, j] = qtop.ry(phi=phi).data[i,j]
    return Qobj(m)

def sz(phi,N):
    m = np.diag(np.zeros(N, dtype=complex))
    for i in range(2):
        for j in range(2):
            m[i, j] = qtop.rz(phi=phi).data[i,j]
    return Qobj(m)



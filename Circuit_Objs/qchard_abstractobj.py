# Author: Long Nguyen, Konstantin Nesterov
###########################################################################
"""The Fluxonium class for representing superconducting fluxonium qubits.
"""

__all__ = ['Fluxonium', 'heavy_fluxonium_params', 'light_fluxonium_params', 'std_fluxonium_sim_params']

import numpy as np
import qutip as qt
import dill
from abc import *

class AbstractQubit(ABC):
    name:str

    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def __str__(self):
        pass
    
    @abstractmethod
    def _save_str(self):
        pass

    @abstractmethod
    def _scale_E_params(self, scaling):
        pass
    
    @abstractmethod
    def levels(self):
        pass

    @abstractmethod
    def n(self):
        pass
    
    @abstractmethod
    def phi(self):
        pass

    # Probably better to just have a class method which correctly dishes out the variable.
    @abstractmethod
    def n_ij(self):
        pass
    
    @abstractmethod
    def phi_ij(self):
        pass

    def transition_energies(self, lower_level=0, nlev=None) -> np.ndarray:
        '''From provided lower level, finds the zeroed transition energy to the upper levels'''
        if nlev is None:
            nlev = self.nlev
        eigvals = self.levels(nlev=nlev)[lower_level:nlev]
        transitions = eigvals - eigvals[0]
        return transitions # TODO: Remove the zeroed energy here. This will be a breaking change.

    def save_obj(self, dir:str):
        with open(dir+self._save_str(), 'wb') as f:
            dill.dump(self, f)


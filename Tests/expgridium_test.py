# Author: Thomas Ersevim, 2026
###################################################################################################

'''An ExpGridium test suite'''

import sys
sys.path.append('/Users/thomasersevim/QNL/2q_gridium')

import numpy as np
import pytest
import matplotlib.pyplot as plt

from Circuit_Objs.qchard_expgridium import *
import Notebooks.plotting_settings

'''
soft_ExpGridium = ExpGridium(**soft_ExpGridium_params,
                             **std_ExpGridium_operation_params,
                             **std_ExpGridium_sim_params)
hard_ExpGridium = ExpGridium(**hard_ExpGridium_params,
                         **std_ExpGridium_operation_params,
                         **std_ExpGridium_sim_params)
'''

def test_spectrum():
    pass

def test_eigenvals():
    pass

# TODO: Convergence Tests (to find optimal/sufficent within error lc values for cphase sims

if __name__=='__main__':
    pass
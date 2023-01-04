# /usr/bin/env python3

"""LOKI (Locally Optimal search after K-step Imitation) is a algorithm for finding the optimal policy for a model"""

# Imports

import numpy as np
import random
import time

# Functions

def random_sample_switcher(d=3, Nmin=25, Nmax=50):
    """Choose a random moment for switching from IL to RL based on a uniform distribution

    The values are chosen between Nmin and Nmax. d, Nmin and Nmax are set from suggestions 
    in the original paper. https://arxiv.org/abs/1805.10413.

    Input:
        d: number of dimensions
        Nmin: minimum number of samples
        Nmax: maximum number of samples
        
    Output:
        K: number of samples before switching from IL to RL
        
    """
    prob_mass = np.arange(Nmin//2, Nmax+1) ** d 
    prob_mass = prob_mass / np.sum(prob_mass)
    switch_iter = np.random.choice(np.arange(Nmin//2, Nmax+1), p=prob_mass)

    return switch_iter
    

# Classes

class LOKI:
    
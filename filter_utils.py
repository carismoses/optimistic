import numpy as np
from numpy.random import uniform, randn

def create_uniform_particles(N, D, ranges):
    '''
    :param N: number of particles
    :param D: number of state dimensions
    :param ranges: list of of length D of (min, max) ranges for each state dimension
    '''
    particles = np.empty((N, D))
    weights = np.ones(N)*(1/N)
    for d in range(D):
        particles[:, d] = uniform(*ranges[d], size=N)
    return particles, weights

def create_gaussian_particles(N, D, means, stds):
    '''
    :param N: number of particles
    :param D: number of state dimensions
    :param means: list of of length D of mean for each state dimension
    :param stds: list of of length D of st dev for each state dimension
    '''
    particles = np.empty((N, D))
    for d in range(D):
        particles[:, d] = means[d] + (randn(N) * stds[d])
    return particles

import sys
import scipy.constants as csts
import numpy as np
import matplotlib.pyplot as plt
from nbody_helpers import *



def get_pot_filter_grav(N,dx,fuzz_dist):
    """
    What I want: a filter the size of the original space in terms of number of cells.
    This filter plots a potential of 1/r where r = 0 is the center of the space.
    The fuzz dist is the distance under which we set the potential do a constant as to avoid huge values.

    This will most-likely be a relatively costly computation, so better reduce the number of times it needs to be done.
    I think it should only be required once at the beginning of the sim, during the  setup.
    """

    x = np.linspace(-(N[0]-1)/2,(N[0]-1)/2,N[0],dtype=float)
    y = np.linspace(-(N[1]-1)/2,(N[1]-1)/2,N[1],dtype=float)
    z = np.linspace(-(N[2]-1)/2,(N[2]-1)/2,N[2],dtype=float)
    #Get grid indicating vector location of every point in space.
    locations = np.array(np.meshgrid(x,y,z))
    locations = np.einsum('ljik->ijkl',locations) #Fix some meshgrid oddity wrt order of dimensions.
    distances = np.sqrt(np.sum((locations*dx)**2,axis=3)) # array representing distance from center of each point in terms of spatial units

    r = np.where(distances>fuzz_dist,distances,fuzz_dist)

    pot_filter = -1/r

    #round to highest degree of precision possible given double accuracy
    max_prec = 16 - np.log10(np.max(np.abs(pot_filter))).astype(int)
    pot_filter = round_floats(pot_filter)

    return pot_filter # Not even once...




def get_potential(dens,pot_filter):
    dens_ft = np.fft.fftn(dens)
    pot_filter_ft = np.fft.fftn(pot_filter)

    pot_ft = np.real(np.fft.ifftshift(np.fft.ifftn(pot_filter_ft*dens_ft)))
    pot_ft = round_floats(pot_ft)

    return pot_ft



def get_grav_forces(pot,dx,pbc):

    forces = -1.0*get_grads_3d(pot,dx,pbc)
    forces = round_floats(forces)

    return forces

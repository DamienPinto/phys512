import sys
import numpy as np
import matplotlib.pyplot as plt



def get_dist_from_center(pos,L,dx=-1):
    # Recenter position #
    cent_pos = pos - L/2
    dist = np.sqrt(np.sum(pos**2))
    if dx != -1: #If the user submitted some dx, means they want a distnace in terms of spatial measures, not just grid indicies
        dist = dist*dx

    return dist
    


def get_grads_3d(dist,dx,pbc): #pbc: Periodic Boundary Conditions: 0 - No, 1 - Yes
    """
    Not 100% what to do here. Thinking simple 2-sided derivative. 
    If no peridoic boundary conditions:
    Set edges to zero. Will handle later. Edges will be places where if the particle is headed there 
    their velocity in that direction will be reversed as if a perfect rebound.
    If periodic boundary consitions: 
    Keep edges. Particle will loop around and so derivatives at edge constructed from cells on other end of box are valid and useful.
    """

    grads = np.zeros([*(dist.shape),3])

    for i in range(3):
        grads[:,:,:,i] = (np.roll(dist,1,axis=i) - np.roll(dist,-1,axis=i))/(2*dx[i])

    if pbc:
        base = np.zeros([*(dist.shape),3])
        base[1:-1,1:-1,1:-1] = grads[1:-1,1:-1,1:-1]
        # grads = np.pad(grads[1:-1,1:-1,1:-1],(1,1,1),'constant',constant_values=(np.zeros(3),np.zeros(3),np.zeros(3)))

    #Round to highest degree of precision possible given data stored in float64 format
    grads = round_floats(grads)

    return grads



def check_bc(x,v,L,pbc):
    if pbc:
        #Particles that are projected to go "too far", i.e. past an edge of the box
        #go too far by an amount x%N. By looping back around they end up at x%N.
        #This works as well with positive or negative values as, for example,
        #-7%10 = 3, which is good for our purposes because a particle that ends up at a -7
        #index will effectively be at the 3rd index or a box of size 10
        #Velocities continue on as normal since no object was encontered.
        x_checked = x%L
        v_checked = v

        #Code for when I didn't know the convenient nature of the modulo of negative numbers
        # sign = np.sign(x_f)
        # Ns = np.array([universe.N]*len(x_f))*(sign-1)*-0.5
        # new_x = x_f%universe.N
        # x_f = np.round(Ns + sign*new_x).astype(np.int)
    else:
        #Take care of bounces causing reflections in velocity

        v_x = np.where(np.logical_or(x[:,0]>L[0],x[:,0]<0),v[:,0]*-1,v[:,0])
        v_y = np.where(np.logical_or(x[:,1]>L[1],x[:,1]<0),v[:,1]*-1,v[:,1])
        v_z = np.where(np.logical_or(x[:,2]>L[2],x[:,2]<0),v[:,2]*-1,v[:,2])
        v_checked = np.stack((v_x,v_y,v_z),axis=1)

        #Take care of particles going "too far" needing their position adjusted
        #Can only handle one bounce per dimension per timestep time-step here
        #but I doubt particles will be going fast enough
        #to traverse the entire space in one timestep. (or at least I hope not)
        x_x = np.where(np.logical_or(x[:,0]>L[0],x[:,0]<0),L[0]-x[:,0]%L[0],x[:,0])
        x_y = np.where(np.logical_or(x[:,1]>L[1],x[:,1]<0),L[1]-x[:,1]%L[1],x[:,1])
        x_z = np.where(np.logical_or(x[:,2]>L[2],x[:,2]<0),L[2]-x[:,2]%L[2],x[:,2])
        x_checked = np.stack((x_x,x_y,x_z),axis=1)

    return x_checked, v_checked



def pstn2idx(pstns,dx,to_int=0):
    idx = np.round(pstns/dx)
    if to_int:
        idx = idx.astype(np.int)
    return idx



def round_floats(arr):
    #Function that rounds arrays to the highest degree of precision they could have
    #given the largest magnitude of their data and assuming it is stored in float64
    #form
    max_prec = 16 - np.log10(np.max(np.abs(arr))+1e-16).astype(int)-1

    rounded_arr = np.round(arr,max_prec)

    return rounded_arr



# def CFS_AVX(xs,vs,fs,ms,dt):
#     A = forces[xs[:,0],xs[:,1],xs[:,2]]/np.repeat(ms.T,3,axis=0).reshape(len(ms),3)
#     V = vs + dt*A
#     x3 = x_i dt*V
#On hold bc sometimes you want to multiply by 0.5dt and dt for the same A_i,V_i and x_i,
#so just wrote it out in full in main CFS function

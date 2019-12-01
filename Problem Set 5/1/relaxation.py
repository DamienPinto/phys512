import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.constants as csts
from heapq import nlargest
import time




def get_params(path):
    param_str = open(path,'r').read()
    params = np.fromstring(param_str,sep=' ',dtype=float)

    return params




def get_3d_cyl(shape,r,l,pos,drctn=0):
    '''
    Function that can create a binary mask for a 3D cylinder of length 'l' and radius 'r' along any of the directions of the primary axes desiganted by 'drctn' and centered at pos. The user is tasked with making sure that the cylinder does not exceed the bounds of the space it will inhabit whose dimensions are specified by 'shape'.

    '''
    def others(drctn):
        '''
        Function that, when given one of three possible indicies specifying an axis in a 3D python array, return the two others.
        '''
        return [(drctn-1)%3,(drctn+1)%3]
    r - np.float64(r)
    drctn = int(drctn)
    [Lx,Ly,Lz] = shape
    Lx = int(Lx)
    Ly = int(Ly)
    Lz = int(Lz)
    x = np.linspace(0,Lx-1,Lx,dtype=float)
    y = np.linspace(0,Ly-1,Ly,dtype=float)
    z = np.linspace(0,Lz-1,Lz,dtype=float)
    #Get grid indicating vector location of every point in space.
    locations = np.array(np.meshgrid(x,y,z))
    locations = np.einsum('ljik->ijkl',locations) #Fix some meshgrid oddity wrt oreder of dimensions.
    res = locations - [[[pos]*Lz]*Ly]*Lx
    #Compute distance from pos along the central axis of the cylinderof every point in space,
    #As well as every point's perpendicular distance from said central axis.
    perp_norm = np.sqrt(np.sum(res[:,:,:,others(drctn)]**2,axis=3))
    par_norm = np.sqrt(res[:,:,:,drctn]**2)

    #Create cylinder mask.
    ones = np.ones(res.shape[:-1])
    zeros = np.zeros(ones.shape)
    cyl = np.where(np.logical_and(perp_norm<=r,par_norm<=l),ones,zeros)
    cyl_mask = np.where(np.logical_and(perp_norm<=r,par_norm<=l),True,False)

    return cyl,cyl_mask,perp_norm




def get_3d_Lapl(data):
    ''
    lapl = (data[0:-2,1:-1,1:-1]+data[2:,1:-1,1:-1]+data[1:-1,0:-2,1:-1]+data[1:-1,2:,1:-1]+data[1:-1,1:-1,0:-2]+data[1:-1,1:-1,2:])/6.0

    return lapl




def get_true_V(lamb,L,l,r,mask):
    V_in = np.ones(r.shape)
    V_out = lamb*np.log(L/2/r)/(2*np.pi*csts.epsilon_0*l)

    V_true = np.zeros(r.shape)
    V_true[mask] = V_in[mask]
    V_true[np.invert(mask)] = V_out[np.invert(mask)]
    return V_true




def main():
    param_filename = sys.argv[1]
    params = get_params(param_filename)
    #             space dims,    r    ,    l    ,  pos    ,  drctn
    cyl_params = [params[:3],params[3],params[4],params[5:8],params[8]]
    R = params[3]
    l = params[4]
    # lamb = params[9]/(2*np.pi*R**2 + l*2*np.pi*R) 
    lamb = 2*np.pi*csts.epsilon_0 #Set charge density such that 
    niter = params[10]
    thresh = params[11]

    cyl,cyl_mask,perp_norm = get_3d_cyl(*cyl_params)
    V = np.zeros(cyl.shape)
    bc = lamb*cyl/(2*np.pi*csts.epsilon_0)
    V = bc.copy()
    V_tmp = np.zeros(V.shape)
    plt.ion()
    plt.imshow(cyl_mask[int(params[0]//2),:,:])

    t1 = time.time()
    for i in range(int(niter)):
        V_tmp[1:-1,1:-1,1:-1] = get_3d_Lapl(V)
        V_tmp[cyl_mask] = bc[cyl_mask]
        diff = np.sum(np.abs(V-V_tmp))
        # plt.clf()
        # plt.imshow(V[int(params[0]//2),:,:])
        # plt.colorbar()
        # plt.pause(0.0001)
        # print(np.all(V==V_tmp))
        # print(np.all(diffs==0.0))
        if diff < thresh:
            print("Things stopped changing substantially after %d steps. Stopping."%(i))
            break
        elif i%200==0:
            print("Average pixel change at step %d is: %f"%(i,diff))
        V[:,:,:] = V_tmp[:,:,:]
    t2 = time.time()
    print("Time taken for relaxation: %f." %(t2-t1))
    
    plt.clf()
    plt.imshow(V[int(params[0]//2),:,:])
    plt.colorbar()
    plt.title("Electric Ptotential Found Numerically")
    plt.savefig("num_V.png")

    final_V = open("final_num_V.txt",'w')
    final_V.write(str(V.flatten().tolist()))
    final_V.close()
    
    rho = V[1:-1,1:-1,1:-1] - get_3d_Lapl(V)
    plt.clf()
    plt.imshow(rho[int(params[0]//2),:,:])
    plt.colorbar()
    plt.title("Charge Density of Charged Cylinder (Relaxation)")
    plt.savefig("rho_relaxation.png")

    rho_non_zero_mask = np.where(rho>0.001,True,False)
    rho_N = np.sum(rho_non_zero_mask)
    rho_tot = np.sum(rho[rho_non_zero_mask])
    rho_avg = rho_tot/rho_N


    print("Average line density is: %f." % (rho_avg))
    print("Top 5 pixels with largest charge densities have values: %s." %(str(nlargest(5,rho.flatten()))))

    V_true = get_true_V(rho_tot,params[0],l,perp_norm,cyl_mask)
    plt.clf()
    plt.imshow(V_true[int(params[0]//2),:,:])
    plt.colorbar()
    plt.title("Analytic Potential as Function of r Only")
    plt.savefig("true_V.png")
    
    diff = np.abs(V-V_true)
    plt.clf()
    plt.imshow(diff[int(params[0]//2),:,:])
    plt.colorbar()
    plt.title("Differences Between Analytic and Numerical Vs")
    plt.savefig("diffs_V.png")



if __name__ == "__main__":
    main()

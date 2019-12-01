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
        Function that, when given one of three possible indicies specifying an axis in a 3D python array, returns the two others.
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
    locations = np.einsum('ljik->ijkl',locations) #Fix some meshgrid oddity wrt order of dimensions.
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





def get_true_V(lamb,R,dist,mask):
    V_in = lamb/(2*np.pi*csts.epsilon_0*R**2)*np.ones(dist.shape)
    V_out = lamb*np.log(R/dist)/(2*np.pi*csts.epsilon_0)

    V_true = np.zeros(dist.shape)
    V_true[mask] = V_in[mask]
    V_true[np.invert(mask)] = V_out[np.invert(mask)]
    
    return V_true




def get_crappy_V(filepath,shape):
    data_flat = np.fromstring(open(filepath,'w').read(), sep=", ",dtype=np.float64)
    data = data_flat.reshape(shape)
    plt.imshow(data[int(len(data)//2)])
    
    return data




def pad_mat(mat,pad_amt):
    mat_shape = np.array(np.shape(mat))
    base_shape = mat_shape+2*pad_amt

    # print("Mat shape: ",mat.shape)

    base = np.zeros(base_shape)
    # print("Base shape: ",base.shape)
    # print("Truncated base shape: ",(base[pad_amt:-pad_amt,pad_amt:-pad_amt,pad_amt:-pad_amt]).shape)
    base[pad_amt:-pad_amt,pad_amt:-pad_amt,pad_amt:-pad_amt] = mat[:,:,:]

    return base




#Basically same as prof Sievers' code in his laplace_conjgrad.py
def conj_grad(data,guess,A_func,mask,niter,thresh):
    r = data - A_func(guess,mask)
    p = r.copy()
    results = guess.copy()
    t1 = time.time()
    relax_thresh = 1.0
    comp = 0

    for i in range(niter):
        Ap = (A_func(pad_mat(p,1),mask)) #1 is actually degree of derivative that is enacted on data -1.
        r2 = np.sum(r*r)
        alpha = r2/np.sum(Ap*p)

        changes = pad_mat(alpha*p,1)
        if i%10==0:
            print("(i,r2): (%d,%.2e)"%(i,r2))
            print("Sum of sqrt of changes is: %.2e." % (np.sum(np.abs(changes))))

        results = results + changes
        if np.sum(np.abs(changes)) < relax_thresh and comp == 0.0:
            print("Sum of magnitude of pixel changes has passed the threshold set during the relaxation method script in %d iterations."%(i))
            plt.clf()
            plt.imshow((results+changes)[int(len(results)//2),:,:])
            plt.colorbar()
            plt.title("Conjugate Gradient and RElaxation Comparison")
            plt.savefig("comp_to_relax_V.png")
            comp = 1

        r_new = r - alpha*Ap
        beta = np.sum(r_new*r_new)/r2
        p = r_new + beta*p
        r = r_new

        plt.clf()
        plt.imshow(results[int(len(results)//2),:,:])
        plt.colorbar()
        plt.pause(0.001)

        if np.sum(r*r) < thresh:
            print("Squared residuals at step %d are %.2e < %.2e. Stopping and returning results." % (i,np.sum(r*r),thresh))
            t2 = time.time()
            print("Time taken for conj grad method: %f." %(t2-t1))
            
            return results
    print("Maximum number of iterations undergone without achieving bar designated by threshold. Returning results with squared residuals %f." % (np.sum(r*r)))
    print("Time taken for conj grad method: %f." %(t2-t1))
    
    return results




#Again, basically same as prof Sievers' code in his laplace_conjgrad.py
def mat_3d_Lapl(data,mask):
    data_new = data.copy()
    data_new[mask]=0
    avg = (data_new[:-2,1:-1,1:-1]+data_new[2:,1:-1,1:-1]+data_new[1:-1,:-2,1:-1]+data_new[1:-1,2:,1:-1]+data_new[1:-1,1:-1,:-2]+data_new[1:-1,1:-1,2:])/6.0
    data_Lapl = avg - data[1:-1,1:-1,1:-1]

    return data_Lapl
    



def main():
    param_filename = sys.argv[1]
    params = get_params(param_filename)
    #             space dims,    r    ,    l    ,  pos    ,  drctn
    cyl_params = [params[:3],params[3],params[4],params[5:8],params[8]]
    R = params[3]
    print(R)
    l = params[4]
    # lamb = params[9]/(2*np.pi*R**2 + l*2*np.pi*R) 
    lamb = 2*np.pi*csts.epsilon_0*R**2 #Set charge density such that V = 1 inside cylinder.
    niter = int(params[10])
    thresh = params[11]
    plt.ion()

    cyl,cyl_mask,perp_norm = get_3d_cyl(*cyl_params)
    #If you want to use what the relaxation solution came up with.
    # crappy_V = get_crappy_V("../1/final_num_V.txt",params[:3])
    V_init = np.zeros(cyl.shape)
    bc = lamb*cyl/(2*np.pi*csts.epsilon_0*R**2)
    #Kind of just copying Prof. Sievers' code here, not sure why he uses this as his data, I guess it's just a way in which our actual data can be modified by some noise or some feature of our data-taking process?
    b = -(bc[:-2,1:-1,1:-1]+bc[2:,1:-1,1:-1]+bc[1:-1,:-2,1:-1]+bc[1:-1,2:,1:-1]+bc[1:-1,1:-1,:-2]+bc[1:-1,1:-1,2:])/6.0
    
    V_cg = conj_grad(b,V_init,mat_3d_Lapl,cyl_mask,niter,thresh)

    plt.clf()
    plt.imshow(V_cg[int(len(V_cg)//2),:,:])
    plt.colorbar()
    plt.title("Electric Potential Obtained with Conjugate Gradient")
    plt.savefig("cg_V.png")

    plt.clf()
    x = np.linspace(0,int(len(V_cg)//2)-1,int(len(V_cg)//2))
    plt.plot(x,V_cg[int(len(V_cg)//2),int(len(V_cg)//2),int(len(V_cg)//2):])
    V_in = np.ones(int(R))
    V_out = np.log(int(len(V_cg)//2)/x[int(R):])*0.6
    V = np.concatenate((V_in,V_out))
    plt.plot(x,V,color='r')
    plt.title("Electric Potential Obtained with Conjugate Gradient (line from center to edge)")
    plt.savefig("cg_V_line.png")

    V_true = get_true_V(lamb,R,perp_norm,cyl_mask)
    plt.clf()
    plt.imshow(V_true[int(len(V_true)//2),:,:])
    plt.colorbar()
    plt.title("Analytic Electric Potential")
    plt.savefig("true_V.png")

    diffs = np.abs(V_cg - V_true)
    plt.clf()
    plt.imshow(diffs[int(len(diffs)//2),:,:])
    plt.colorbar()
    plt.title("Differences in Electric Potentials")
    plt.savefig("diffs_V.png")

    rho = mat_3d_Lapl(V_cg,cyl_mask)
    plt.clf()
    plt.imshow(rho[int(len(rho)//2),:,:])
    plt.colorbar()
    plt.title("Charge Density Obtained from Conjugate Gradient")

    

    
if __name__ == "__main__":
    main()

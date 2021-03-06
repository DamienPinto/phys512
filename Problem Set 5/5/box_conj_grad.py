import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.constants as csts
from scipy.interpolate import RegularGridInterpolator
from heapq import nlargest
import time




def get_params(path):
    param_str = open(path,'r').read()
    params = np.fromstring(param_str,sep=' ',dtype=float)

    return params




def get_3d_box(shape,l,pos):
    print(shape)
    def others(drctn):
        '''
        Function that, when given one of three possible indicies specifying an axis in a 3D python array, returns the two others.
        '''
        return [(drctn-1)%3,(drctn+1)%3]

    [Lx,Ly,Lz] = shape
    Lx = int(Lx)
    Ly = int(Ly)
    Lz = int(Lz)

    x = np.linspace(0,Lx-1,Lx,dtype=float)
    y = np.linspace(0,Ly-1,Ly,dtype=float)
    z = np.linspace(0,Lz-1,Lz,dtype=float)

    locations = np.array(np.meshgrid(x,y,z))
    locations = np.einsum('ljik->ijkl',locations) #Fix some meshgrid oddity wrt order of dimensions.
    res = locations - [[[pos]*Lz]*Ly]*Lx
    #Compute distance from pos along the central axis of the cylinderof every point in space,

    # x_faces = np.where(np.logical_and(np.abs(res[:,:,:,0])==l/2, np.all(np.abs(res[:,:,:,others(0)])<=l/2)),1.0,0.0)
    # y_faces = np.where(np.logical_and(np.abs(res[:,:,:,1])==l/2, np.all(np.abs(res[:,:,:,others(1)])<=l/2)),1.0,0.0)
    # z_faces = np.where(np.logical_and(np.abs(res[:,:,:,2])==l/2, np.all(np.abs(res[:,:,:,others(2)])<=l/2)),1.0,0.0)

    # cube = x_faces+y_faces+z_faces
    cube = np.zeros(shape.astype(int))
    cube[int(pos[0]-l/2):int(pos[0]+l/2),int(pos[1]-l/2):int(pos[1]+l/2),int(pos[2]-l/2):int(pos[2]+l/2)] = np.ones((int(l),int(l),int(l)))
    print(np.all(cube==np.zeros(cube.shape)))
    plt.clf()
    plt.imshow(cube[:,:,len(cube)//2])
    plt.pause(3)
    cube_mask = np.where(cube==1.0,True,False)

    return cube,cube_mask,res




# def get_true_V(lamb,R,dist,mask):
#     V_in = lamb/(2*np.pi*csts.epsilon_0*R**2)*np.ones(dist.shape)
#     V_out = lamb*np.log(R/dist)/(2*np.pi*csts.epsilon_0)

#     V_true = np.zeros(dist.shape)
#     V_true[mask] = V_in[mask]
#     V_true[np.invert(mask)] = V_out[np.invert(mask)]
    
#     return V_true




# def get_crappy_V(filepath,shape):
#     data_flat = np.fromstring(open(filepath,'w').read(), sep=", ",dtype=np.float64)
#     data = data_flat.reshape(shape)
#     plt.imshow(data[int(len(data)//2)])
    
#     return data




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
def conj_grad(data,guess,A_func,mask,side,niter,thresh):
    r = data - A_func(guess,mask) #Our data minus our best guess at a fit.
    p = r.copy()
    results = guess.copy()
    t1 = time.time()
    relax_thresh = 1.0
    comp = 0

    for i in range(niter):
        results = results+side
        Ap = (A_func(pad_mat(p,1),mask)) #1 is actually degree of derivative that is enacted on data -1.
        r2 = np.sum(r*r)
        alpha = r2/np.sum(Ap*p) #Magnitude squared of our residuals divided by the magnitude of our gradient vector?

        changes = pad_mat(alpha*p,1)
        if i%10==0:
            print("(i,r2): (%d,%.2e)"%(i,r2))
            print("Sum of sqrt of changes is: %.2e." % (np.sum(np.abs(changes))))

        results = results + changes

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




#This a copy-paste essentially of prof Sievers' function as I don't
#100% understand how it works. He says in his comment that, in charge
#free region, "we know that [...] V0+0.25*(V_l+V_r+V_u+V_d)=0"
#But it seems to be that, from Laplace's, we know that
# V0 = 0.25*(V_l+V_r+V_u+V_d)
#Essentially the sae thing but with a sign-difference
# def make_rhs(data,mask):
#     rhs = np.zeros(data.shape)
#     rhs[:-1,:,:] = rhs[:-1,:,:] + data[1:,:,:]
#     rhs[1:,:,:] = rhs[1:,:] + data[:-1,:,:]
#     rhs[:,:-1,:] = rhs[:,:-1,:] + data[:,1:,:]
#     rhs[:,1:,:] = rhs[:,1:,:] + data[:,:-1,:]
#     rhs[:,:,:-1] = rhs[:,:,:-1] + data[:,:,1:]
#     rhs[:,:,1:] = rhs[:,:,1:] + data[:,:,-1:]
#     rhs[mask] = 0.0

#     return rhs[1:-1,1:-1,1:-1]



def make_rhs(data):
    rhs = -(data[2:,1:-1,1:-1]+data[:-2,1:-1,1:-1]+data[1:-1,:-2,1:-1]+data[1:-1,2:,1:-1]+data[1:-1,1:-1,2:]+data[1:-1,1:-1,:-2])/6.0
    return rhs




#Again, basically same as prof Sievers' code in his laplace_conjgrad.py
def mat_3d_Lapl(data,mask):
    data_new = np.array(data.copy())
    # print(data.shape,mask.shape)
    data_new[mask] = 0.0
    avg = (data_new[:-2,1:-1,1:-1]+data_new[2:,1:-1,1:-1]+data_new[1:-1,:-2,1:-1]+data_new[1:-1,2:,1:-1]+data_new[1:-1,1:-1,:-2]+data_new[1:-1,1:-1,2:])/6.0
    data_Lapl = avg - data[1:-1,1:-1,1:-1]

    return data_Lapl



# Again, basically Sievers' de_res, but returns all resolutions at once.
def de_res(data,f,n):
    all_res = [None]*n
    all_res[0] = data
    deres_shape = np.array(data.shape)
    for i in range(1,n):
        deres_shape = deres_shape//int(f)
        # print("deres_shape: ",deres_shape)
        low_res = np.zeros(deres_shape,dtype=data.dtype)
        F = f**i
        low_res = np.maximum(low_res,data[::F,::F,::F])
        low_res = np.maximum(low_res,data[(F-1)::F,::F,::F])
        low_res = np.maximum(low_res,data[::F,(F-1)::F,::F])
        low_res = np.maximum(low_res,data[::F,::F,(F-1)::F])
        low_res = np.maximum(low_res,data[::F,(F-1)::F,(F-1)::F])
        low_res = np.maximum(low_res,data[(F-1)::F,(F-1)::F,::F])
        low_res = np.maximum(low_res,data[(F-1)::F,::F,(F-1)::F])
        low_res = np.maximum(low_res,data[(F-1)::F,(F-1)::F,(F-1)::F])
        

        all_res[i] = low_res

    return all_res




def up_res_conj_grad(bc_scales,side_scales,mask_scales,A_func,rhs_func,f,niter,thresh):
    n = np.shape(mask_scales)[0]
    
    rhs_scales = [None]*n
    rhs_scales[-1] = rhs_func(bc_scales[-1])#,mask_scales[-1])
    
    init_x = 0*mask_scales[-1]
    x_scales = [None]*n
    # print(np.shape(rhs_scales[-1]),np.shape(init_x),np.shape(mask_scales[-1]))
    x_scales[-1] = conj_grad(rhs_scales[-1],init_x,A_func,mask_scales[-1],side_scales[-1],niter,thresh)

    #Do the thing
    for i in range(n-2,-1,-1):
        rhs_scales[i] = rhs_func(bc_scales[i])#,mask_scales[i])
        x_tmp = up_res(x_scales[i+1],f,1)[0]
        x_scales[i] = conj_grad(rhs_scales[i],x_tmp,A_func,mask_scales[i],side_scales[i],niter,thresh)
        print(x_scales[i].shape)
        plt.clf()
        plt.imshow(x_scales[i][int(len(x_scales[i])//4),:,:])
        plt.colorbar()
        plt.pause(0.01)

    #Paste back in bcs bc/ we were cheeky and didn't consider then while fitting. Hope this was legal.
    for i in range(n):
        x_scales[i][mask_scales[i]] = bc_scales[i][mask_scales[i]]

    return x_scales



def up_res(data,f,n):
    all_res = [None]*n
    og_shape = np.array(data.shape)
    upres_shape = og_shape.copy()
    x = np.linspace(1,og_shape[0],upres_shape[0])
    y = np.linspace(1,og_shape[1],upres_shape[1])
    z = np.linspace(1,og_shape[2],upres_shape[2])
    interp_func = RegularGridInterpolator((x,y,z),data)

    for i in range(n):
        upres_shape = np.array(upres_shape*int(f),dtype=int)
        xf = np.linspace(1,og_shape[0],upres_shape[0])
        yf = np.linspace(1,og_shape[1],upres_shape[1])
        zf = np.linspace(1,og_shape[2],upres_shape[2])

        grid = np.array(np.meshgrid(xf,yf,zf,indexing='ij'))
        grid = np.einsum('lijk->ijkl',grid)
        all_res[i] = interp_func(grid)

    return all_res
        
        



def main():
    param_filename = sys.argv[1]
    params = get_params(param_filename)
    #             space dims,    l    ,  pos
    cube_params = [params[:3],params[3],params[4:7]]
    l = params[3]
    niter = int(params[7])
    thresh = params[8]
    plt.ion()

    ################################################
    # Get boundary conditions and mask at full res #
    ################################################
    
    
    #If you want to use what the relaxation solution came up with.
    # crappy_V = get_crappy_V("../1/final_num_V.txt",params[:3])
    T_init = np.zeros(cube_params[0].astype(int))
    # bc = lamb*cyl/(2*np.pi*csts.epsilon_0*R**2)
    ##bc now becomes side we want to heat
    print(cube_params[0])
    cube,cube_mask,res = get_3d_box(cube_params[0],l,cube_params[2])
    side = np.where(np.logical_and(cube==1.0,res[:,:,:,0]==-l/2),1.0,0.0)
    print(np.sum(side)) #= l*l
    #####################################################
    # De-res a few times by factor of two bcs and masks #
    #####################################################

    #Doing like prof Sievers and de-res-ing 6 times before starting
    npass = 3
    de_res_factor = 2
    bc_scales = de_res(cube,de_res_factor,npass)
    side_scales = de_res(side,de_res_factor,npass)
    mask_scales = de_res(cube_mask,de_res_factor,npass)

    
    #############################################################
    # Initialize rhs and x for lowest resolution mask and bc    #
    # Cycle through scales while effecting a conjugate gradient #
    #############################################################

    T_scales = up_res_conj_grad(bc_scales,side_scales,mask_scales,mat_3d_Lapl,make_rhs,2,niter,thresh)

    plt.clf()
    plt.imshow(T_scales[0][:,128,:])
    plt.colorbar()
    plt.title("Variable Resolution Conjugate Gradient on Heated Box")
    plt.savefig("var_res_conj_grad_T.png")
    plt.show()
    
    T_line = T_scales[0][128:256,128,128]
    p = np.arange(128,int(128+len(T_line)))
    plt.clf()
    plt.plot(p,T_line)
    plt.title("Temperature Through Heated Box")
    plt.xlabel("Position")
    plt.ylabel("T")
    plt.savefig("line_T.png")

    

    
if __name__ == "__main__":
    main()

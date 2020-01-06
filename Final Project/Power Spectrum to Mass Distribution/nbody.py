import numpy as np
import sys
import matplotlib.pyplot as plt
from nbody_phys import *
from nbody_helpers import *
from p_spec_tools import *
from power_spectrum_to_universe import *
from scipy.signal import convolve as conv



##############################################################################
# Commenting this class-based code out in favor of info contained in         #
# arrays found in Space class. This allows for easier updating en-masse.     #
# The structure below as well as functions related to it in the Space class  #
# can be uncommented. Be sure to comment-out array-related equivalents       #
##############################################################################

#
# class Particle:
#     def __init__(self,m=1,pos=[0,0,0],v=[0,0,0]):
#         self.m = m
#         self.pos = pos
#         self.v = v

#         def copy(self):
#             return Particle(self.m,self.pos,self.v)



class Space:
    #This config of inputs makes it quite easy to input a particle distribution
    #corresponding to certain situations, just have to compute the init values
    #for said situations and feed them to the space beforehand.
    def __init__(self,L=[100,100,100],dx=[1,1,1],dt=1,n=1,init_type=0,m_dist=None,pos_dist=None,v_dist=None,fuzz_dist=1):
        self.L = L
        self.dx = dx
        self.N = (np.round(L/dx)).astype(np.int)
        print("N: ",self.N)

        self.dt = dt

        self.num_prtcl = n
        self.particles = [None]*n

        #Function for #1 to see if lone particle stays still.
        if init_type == 1:
            print("Initializing space with one particle")
            # pos1 = self.L/2
            pos1 = np.array([self.L[0]/2,self.L[1]/4,49.5])
            print("pos1: ",pos1)
            self.pstns = np.array([pos1])
            # self.pstns = np.array([[49.7,0.0,0.0]])
            # print(self.pstns[:,0])
            self.ms = np.array([1.0])
            self.vs = np.array([[0.0,10.0,20.0]])

            # self.particles[0] = Particle()
            # return

        #Function for rest of numbers. Nedd to compute situations corresponding to "stable" orbit.
        elif init_type == 0:
            print("Initializing particles in space according to some specified distribution")
            if np.all([len(m_dist),len(pos_dist),len(v_dist)]==self.num_prtcl):
                print("Not all dimensions to input distributions parameter arrays match.")
                print("You should revise their generation.")
                print("Quiting.")
                quit()
            # for i in range(self.num_prtcl):
            #     self.particles[i] = Particle(m_dist[i],pos_dist[i],v_dist[i])
            self.pstns = pos_dist
            self.ms = m_dist
            self.vs = v_dist
            # self.particles = np.stack((m_dist,pos_dist,v_dist),axis=1) # if for some reason you want to access
            # the info of one specific particle at a time, but that kind of defeats the purpose of the Fourier method.

        #Code for orbit. Chose 2 for obvious reasons
        elif init_type == 2:
            print("Initializing particles in orbit around each other.")
            self.pstns = np.array([[25.0,27.5,25.0],[25.0,22.5,25.0]])
            self.ms = np.array([10.0,10.0])
            self.vs = np.array([[0.0,0.0,np.sqrt(0.5/5)],[0.0,0.0,-np.sqrt(0.5/5)]])
            # self.vs = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0]])

        rho = np.zeros(self.N).astype(np.float64)
        # for i in range(self.num_prtcl):
        #Convert positions to idex-array and feed that to rho
        pos_idx = pstn2idx(self.pstns,self.dx,1)
        print("What I'm using as indices: ",[pos_idx[:,0],pos_idx[:,1],pos_idx[:,2]])
        print("In universe, pos_idx.shape: ", np.shape(pos_idx))
        print("In universe, ms.shape all 0.0?: ",np.all(self.ms==0.0))
        print("In universe, mass, position, and index position of first particle: %f, %s, %s"%(self.ms[0],str(self.pstns[0].tolist()),str(pos_idx[0].tolist())))
        rho[tuple(pos_idx.T)] += 1.0*self.ms
        print("In universe, rho_mid: ",rho[len(rho)//2,len(rho)//2,len(rho)//2])
        self.rho = rho

        #Highest degrees of precision possible given double precision for positions and accelerations
        #given conservative values (i.e. considering the highest possible distance possible and highest possible force computed).
        self.pstn_prec = np.log10(L).astype(int) - 16
        self.acc_prec = np.log10((np.max(self.ms)/np.min(self.ms))*dx/(fuzz_dist*(fuzz_dist-np.min(dx)))).astype(int) - 16


    def update_rho(self):
        rho = np.zeros(self.N).astype(np.float64)
        pos_idx = pstn2idx(self.pstns,self.dx,1)
        for i in range(self.num_prtcl):
            rho[tuple([pos_idx[i,0],pos_idx[i,1],pos_idx[i,2]])] = 1.0*self.ms[i]
        self.rho = rho

        return


    # def update_prtcl_pos(self,new_pos):
    #     for i in range(self.num_prtcl):
    #         self.particles[i].pos = new_pos[i]
    #     return

    # def update_prtcl_v(self,new_v):
    #     for i in range(self.num_prtcl):
    #         self.particles[i].v = new_v
    #     return
        
    def update_prtcl_pos_n_vs(self,new_pstns,new_vs):
        self.pstns = new_pstns
        self.vs = new_vs
        self.update_rho()

        return



def evolveCFS(universe,pbc,get_forces,get_pot,pot_filter):
    '''
    Implementation of "Classical Fourth-order Scheme", or "classic Runge-Kutta method"
    y_(t+h) = y_t + (h/6)*(Y_1' + 2*Y_2' + 2*Y_3' + Y_4')
    y_t'(t) = g(t,y_t)

    Y_1' = g(t, y_t)
    Y_2' = g(t+h/2, y_t + (h/2)*Y_1')
    Y_3' = g(t+h/2, y_t + (h/2)*Y_2')
    Y_4' = g(t+h, y_t + h*Y_3')

    Supposed to give errors of order O(h^4) since using Simpson rule of order 3

    Have to do this with both the position where y->pos & Y->v, 
    and for the velocities, where y->v & Y->a->forces(pos)/m
    In both cases h->dt
    '''

    def get_AVX(universe,V_prev,dt,pbc,get_forces,get_pot,pot_filter):
        #Get projection of all particles' positiond dt in the future based on their current velocities
        dx = universe.dx
        N = universe.N
        L = universe.L
        ms = universe.ms
        x = universe.pstns
        v = universe.vs
        
        x_new = x + V_prev*dt
        # print("x_new: ",x_new)
        x_new = round_floats(x_new)
        
        #Check periodic boundary conditions and modify xs accoringly
        # print("x before check: ",x_new)
        x_new_checked, V_prev_checked = check_bc(x_new,V_prev,L,pbc) 
        # print("x after check: ",x_new_checked)

        #Now that we have evolved the system a time-step, check new potential caused by new distribution and new forces
        #present to get A_new
        rho_new_checked = np.zeros(N)
        # print("Shape of un-indexed x_new_checked: ",x_new_checked.shape
        pos_idx_x_checked = pstn2idx(x_new_checked,dx,1)
        for i in range(len(x_new_checked)):
            # print("x_new_checked shape: ",x_new_checked.shape)
            rho_new_checked[pos_idx_x_checked[i,0],pos_idx_x_checked[i,1],pos_idx_x_checked[i,2]] += ms[i]
        
        pot_new_checked = get_pot(rho_new_checked,pot_filter)
        f_new_checked = get_forces(pot_new_checked,dx,pbc)
        # A_new_checked = f_new_checked[pos_idx_x_checked[:,0],pos_idx_x_checked[:,1],pos_idx_x_checked[:,2]]/np.repeat(ms.T,3,axis=0).reshape(len(ms),3)
        A_new_checked = f_new_checked[tuple(pos_idx_x_checked.T)]/np.repeat(ms.T,3,axis=0).reshape(len(ms),3)

        V_new = V_prev_checked + A_new_checked*dt
        # print("V_prev_checked: ",V_prev_checked)
        # print("A_new_checked: ",A_new_checked)
        mod = np.where(V_prev==V_prev_checked,1,-1) #Array that is -1 on axis of reflection, 1 in every other. If no reflection, just 1.
        mV_new = V_new*mod #New velocity but in the same direction as the initial velocity before potential reflection.
        # print("x at end of get_AVX: ",x)
        # print("mV_new at end of AVX: ",mV_new)
        return A_new_checked,V_new,mV_new

    dx = universe.dx
    dt = universe.dt

    x_i = universe.pstns
    v_i = universe.vs
    ms = universe.ms
    N = universe.N
    L = universe.L
    pot = get_pot(universe.rho,pot_filter)
    forces = get_grav_forces(pot,dx,pbc)
    pos_idx_x_i = pstn2idx(x_i,dx,1)
    a_i = forces[pos_idx_x_i[:,0],pos_idx_x_i[:,1],pos_idx_x_i[:,2]]/np.repeat(ms.T,3,axis=0).reshape(len(ms),3)

    #Get energy of system bc it's convenient
    E_pot = np.sum(-ms*pot[pos_idx_x_i[:,0],pos_idx_x_i[:,1],pos_idx_x_i[:,2]])#/np.array([np.max(L)]*universe.num_prtcl))
    E_kin = np.sum(0.5*ms*np.sum(v_i**2,axis=1))
    E_pot = round_floats(E_pot)
    E_kin = round_floats(E_kin)

    # x2 = x_i + dt*v_i
    # x2 = round_floats(x1)
    # rx1, rV1 = check_bc(x1,v_i,L,pbc)
    A1 = a_i
    V1 = v_i
    # print("Position before AVX2: ",universe.pstns)
    #                                IMPORTANT, will not be factor of 0.5 at last step
    A2,V2,mV2 = get_AVX(universe,v_i,0.5*dt,pbc,get_forces,get_pot,pot_filter)
    # print("Position after AVX2: ",universe.pstns)

    A3,V3,mV3 = get_AVX(universe,mV2,0.5*dt,pbc,get_forces,get_pot,pot_filter)
    # print("##### 4 Starting ####")
    A4,V4,mV4 = get_AVX(universe,mV3,dt,pbc,get_forces,get_pot,pot_filter)
    # print("As: ",A1,A2,A3,A4)
    # print("Vs: ",V1,V2,V3,V4)
    # print("mV4: ", mV4)

    # print("x_i: %s, v_i: %s" % (str(x_i),str(v_i)))

    x_f = x_i + (dt/6)*(V1 + 2*mV2 + 2*mV3 + mV4)
    v_f = v_i + (dt/6)*(A1 + 2*A2 + 2*A3 + A4)

    # print("Final position b4 check: ",x_f)
    # print("Final velocity b4 check: ",v_f)

    x_f,v_f = check_bc(x_f,v_f,L,pbc)

    x_f = round_floats(x_f)
    v_f = round_floats(v_f)

    ## PRINT EVERYTHING ##
    # print("x_f: %s, v_f: %s" % (str(x_f),str(v_f)))
    # print("A1: ",A1)
    # print("A2: ",A2)
    # print("A3: ",A3)
    # print("A4: ",A4)
    # print("V1: ",V1)
    # print("V2: ",V2)
    # print("V3: ",V3)
    # print("V4: ",V4)
    # print("mV1: ",mV1)
    # print("mV2: ",mV2)
    # print("mV3: ",mV3)
    # print("mV4: ",mV4)    


    # print("Final position after check: ",x_f)
    # print("Final velocity after check: ",v_f)

    
    universe.update_prtcl_pos_n_vs(x_f,v_f)
    
    return E_pot,E_kin,pot
        


def evolve(universe,fuzz_dist,niter,pbc):
    # So you've got a universe, watch'ya gonna do with it? #

    # Need to iterate through it. #

    # Need to compute an initial potential first. #
    N = universe.N
    dx = universe.dx
    pot_filter = get_pot_filter_grav(N,dx,fuzz_dist)
    pot = get_potential(universe.rho,pot_filter)

    E_pot_log = []
    E_kin_log = []
    E_tot_log = []
    # plt.ion()
    plt.clf()
    fig, axes = plt.subplots(1,2)
    fig.suptitle("Real-Time Simulation Density")
    axes[0].axis('off')
    axes[1].axis('off')

    for i in range(niter):
        print("Step %d of %d." % (i+1,niter))
        pstns_idx = pstn2idx(universe.pstns,dx,1)
        im1 = axes[0].imshow(universe.rho[len(universe.rho)//2,:,:])
        im2 = axes[1].imshow(pot[len(universe.rho)//2,:,:])
        cb1 = fig.colorbar(im1,ax=axes[0])
        cb2 = fig.colorbar(im2,ax=axes[1])                
        plt.savefig("Many Particles Reflective Boundary Conditions/rbc_snap_" + str(i) + ".png")
        plt.pause(1e-3)
        cb1.remove()
        cb2.remove()
        E_pot, E_kin, pot = evolveCFS(universe,pbc,get_grav_forces,get_potential,pot_filter)
        print("\n\n")
        print("E_pot: ",E_pot)
        print("E_kin: ",E_kin)        
        E_pot_log.append(E_pot)
        E_kin_log.append(E_kin)
        E_tot_log.append(E_kin+E_pot)
        # print("Position: ",universe.pstns)
        # print("Velocity: ",universe.vs)
        

    plt.clf()
    plt.plot(np.arange(niter),E_pot_log,color='b',label="Potential Energy")
    plt.plot(np.arange(niter),E_kin_log,color='r',label="Kinetic Energy")
    plt.plot(np.arange(niter),E_tot_log,color='k',label="Total Energy")
    plt.title("Energy of System During Simulation")
    plt.xlabel("Iteration Step")
    plt.ylabel("E")
    plt.legend()
    plt.savefig("E_plot.png")

    return

        
def get_params(param_file):
    params_str = open(params_str,'r').read()
    params = np.fromstring(param_str,sep=" ",dtype=np.float64)

    return params



def main():
    if len(sys.argv) != 1:
        param_file = sys.argv[1]
        params = get_params(param_file)
        if len(params) == 3:
            fuzz_dist = params[-3]
            niter = params[-2]
            pbc = params[-1]
        else:
            uni_params = params[:-3]
            fuzz_dist = params[-3]
            niter = params[-2]
            pbc = params[-1]
        
    else:
        def decomposition(num):
            while num>0:
                m = np.random.random()*10
                if num-m>0:
                    yield m
                    num -= m
                else:
                    yield num
                    num -= num
        L = 64
        dx = 4
        N = int(np.round(L/dx))
        dt = 0.01
        #Stuff useless for this number
        # n = 100000
        # m_dist = np.random.random_sample(n)*100
        # v_dist = np.random.random_sample((n,3))
        # pstn_dist = np.random.random_sample((n,3))*L
        # #       L,dx,dt,n,init_type 
        # uni_params = [np.array([L,L,L]),np.array([dx,dx,dx]),dt,n,0]
        # decomp_test = list(decomposition(50))
        # print("decomp_test: ",decomp_test)


        #######################################################################
        # Construct Desired Power Spectrum and Distribution that Respects It. #
        #######################################################################
        
        k_axis, k_box = get_k_axis(np.array([N,N,N]),dx) #Get k-axis data specified by space dimensions
        p_spec = get_1D_gauss(k_axis,0.25,0.15,6e5) + 1e6*np.exp(-20*k_axis) #Power spectrum being used
        print("Generated p_spec, Making Fourier Space...")
        four_space = make_fourier_space(p_spec,np.array([N,N,N]),dx) #Fourier space generated concordant with power spectrum and that will produce pure values when inverse fourier transformed
        print("Made Fourier Space, Generating Density Distribution...")
        dens_dist = np.real(np.fft.ifftn((four_space), norm="ortho")) #density distribution to be used
        dens_dist -= np.min(dens_dist) #Make sure everything is positive because negative mass densities don't make sense
        dens_dist += 1e-6 #Offset a little bit to have no 0s because the way I decompose the density into masses doesn't know how to handle that
        print("Density Distribution Generated.")

        
        ######################################
        # Plot out all info for examination. #
        ######################################
        
        plt.clf()
        k_axis,dens_dist_p_spec,dens_dist_errs = make_power_spectrum(np.fft.fftshift(np.fft.fftn(dens_dist, norm="ortho")),dx)
        plt.errorbar(k_axis[1:],dens_dist_p_spec[1:],yerr=dens_dist_errs[1:],ecolor='r')
        plt.title("Power Spectrum of Initialized Distribution")
        plt.xlabel(r"k[$\frac{kg}{m^3}$]")
        plt.ylabel("P(k)")        
        plt.savefig("initialized_dist_p_spec.png")
        plt.show()
        # assert(1==0)

        #Smooth out power spectrum because the way I do it gives values for very specific and sometimes similar k-values leading to a lot on unnecesary noise.
        smth_fctr = 5
        remainder = -(len(k_axis[1:])%smth_fctr)
        k_axis_smth = np.concatenate((k_axis[1:remainder].reshape(-1,smth_fctr).mean(axis=1),[np.mean(k_axis[remainder:])]))
        dens_dist_p_spec_smth = np.concatenate((dens_dist_p_spec[1:remainder].reshape(-1,smth_fctr).mean(axis=1),[np.mean(dens_dist_p_spec[remainder:])]))
        dens_dist_errs_smth = np.concatenate((dens_dist_errs[1:remainder].reshape(-1,smth_fctr).mean(axis=1),[np.mean(dens_dist_errs[remainder:])]))        
        
        plt.clf()
        plt.errorbar(k_axis_smth,dens_dist_p_spec_smth,yerr=dens_dist_errs_smth,ecolor='r')
        plt.title("Averaged, ePower Spectrum of Initialized Distribution")
        plt.xlabel(r"k[$\frac{kg}{m^3}$]")
        plt.ylabel("P(k)")        
        plt.savefig("initialized_dist_p_spec_smthd.png")
        plt.show()


        plt.clf()
        plt.plot(k_axis,p_spec)
        plt.title("Initial Power Spectrum")
        plt.xlabel(r"k[$\frac{kg}{m^3}$]")
        plt.ylabel("P(k)")
        plt.savefig("init_p_spec.png")

        plt.clf()
        plt.imshow(dens_dist[len(dens_dist)//2,:,:])
        plt.title("Initial Density Distribution Slice")
        plt.savefig("init_rho_slice.png")

        
        ########################################################################################
        # Decompose Density Distribution into specific masses with specific positions.         #
        # I Haven't found a way of doing this efficiently so for now I use a very long loop... #
        ########################################################################################
        
        m_dist = np.array([])
        pstn_dist = np.array([[None]*3])
        print("Generating Mass and Position Distributions...")
        
        total = np.prod(np.shape(dens_dist))
        for i in range(len(dens_dist)):
            for j in range(len(dens_dist[0])):
                if j==0:
                    sys.stdout.write('\rDealt with %d points of the density field out of %d'%(i*(j+1),total))
                    sys.stdout.flush()
                for k in range(len(dens_dist[0,0])):
                    # print("dens_dist[i,j,k]: ",dens_dist[i,j,k])
                    # if (i*j*k)%(total//1e4) == 0:
                    #     sys.stdout.write('\rDealt with %d points of the density field out of %d'%(i*j*k,total))
                    #     sys.stdout.flush()
                    masses = np.array(list(decomposition(dens_dist[i,j,k])))
                    pstn_entry = np.array([[i*dx,j*dx,k*dx]]*len(masses))

                    # print("masses: ",masses)
                    # print("pstn_entry: ",pstn_entry)
                    # print("pstn_dist: ",pstn_dist)
                    # assert(1==0)
                    m_dist = np.concatenate((m_dist,masses))
                    try:
                        pstn_dist = np.concatenate((pstn_dist,pstn_entry))
                    except:
                        print("Something happened at [i,j,k]=%s"%(str([i,j,k])))
                        print("dens_dist[%d,%d,%d]=%f"%(i,j,k,dens_dist[i,j,k]))
                        print("pstn_entry: ",pstn_entry)
                        print("pstn_dist shape: ",np.shape(pstn_dist))
        print("pstn_dist[0]:",pstn_dist[0])
        pstn_dist = pstn_dist[1:]
        print("First position: ",pstn_dist[0])
        print("First mass: ",m_dist[0])
        print("...Done.")
        print("Shape of m_dist: ",np.shape(m_dist))
        # print("pstn_dist: %s"%str(pstn_dist))
        # print("m_dist: %s" %str(m_dist))
        assert(len(pstn_dist)==len(m_dist))

        n = len(m_dist)
        
        fuzzy_dist = 1
        niter = 350
        pbc = 0
        v_dist = np.zeros((len(masses),3))
        uni_params = [np.array([L,L,L]),np.array([dx,dx,dx]),dt,n,0]
    print("Initializing Space...")
    universe = Space(*uni_params,m_dist=m_dist,v_dist=v_dist,pos_dist=pstn_dist,fuzz_dist=fuzzy_dist)
    print("...Done.")

    uni_rho_ft = np.fft.fftshift(np.fft.fftn(universe.rho,norm="ortho"))
    k_axis,uni_p_spec,errs = make_power_spectrum(uni_rho_ft,dx)
    plt.clf()
    plt.errorbar(k_axis[1:],uni_p_spec[1:],yerr=errs[1:],ecolor='r')
    plt.title("Universe Density P(k)")
    plt.savefig("uni_rho_p_spec.png")
    plt.show()

    remainder = -(len(k_axis[1:])%smth_fctr)
    k_axis_smth = np.concatenate((k_axis[1:remainder].reshape(-1,smth_fctr).mean(axis=1),[np.mean(k_axis[remainder:])]))
    uni_p_spec_smth = np.concatenate((uni_p_spec[1:remainder].reshape(-1,smth_fctr).mean(axis=1),[np.mean(uni_p_spec[remainder:])]))
    errs_smth = np.concatenate((errs[1:remainder].reshape(-1,smth_fctr).mean(axis=1),[np.mean(errs[remainder:])]))

    plt.clf()
    plt.errorbar(k_axis_smth,uni_p_spec_smth,yerr=errs_smth,ecolor='r')
    plt.title("Universe Density P(k)")
    plt.savefig("uni_rho_p_spec_smthd.png")
    plt.show()


    # print("universe.rho max and min: ",np.max(universe.rho),np.min(universe.rho))
    print("universe.rho all 0.0?: ",np.all(universe.rho[len(universe.rho)//2,:,:]==0))
    print("universe.rho==dens_dist?: ",np.all(universe.rho==dens_dist))

    plt.clf()
    plt.imshow(universe.rho[len(universe.rho)//2,:,:])
    plt.title("First Density in Universe Class")
    plt.savefig("uni_rho_check.png")
    plt.show()
    assert(1==0)

    print("Starting Simulation...")
    evolve(universe,fuzzy_dist,niter,pbc)


if __name__ == "__main__":
    main()

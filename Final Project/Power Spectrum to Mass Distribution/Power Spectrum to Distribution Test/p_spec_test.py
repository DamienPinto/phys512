import numpy as np
import matplotlib.pyplot as plt


def make_power_spectrum(f_dist,dx):
    #Function expects fourier space with dc element in center, so shifted.
    
    #Going to assume dist is always cubic
    # f_dist = f_dist/np.sqrt(np.prod(f_dist.shape))
    dims = len(f_dist.shape)
    center = np.array(f_dist.shape)//2
    fs = []
    for n in range(dims):
        fs.append(np.fft.fftshift(np.fft.fftfreq(f_dist.shape[n]))/dx)
    fs = np.array(fs)
    raw_ks = fs*2*np.pi

    #Get box with the shape of the given dist with the k-value of each cell
    if dims != 1:
        raw_k_box = np.array(np.meshgrid(*raw_ks))
        if dims == 2:
            raw_k_box = np.einsum('kji->ijk',raw_k_box)
        elif dims == 3:
            raw_k_box = np.einsum('ljik->ijkl',raw_k_box)
        k_box = np.sqrt(np.sum(raw_k_box**2,axis=dims)) #Box of the wavenumber of every position at that position
        k_axis = np.array(sorted(set(np.abs(k_box).flatten()))) #List of unique k-values in fourier space
    else:
        k_box = np.array(sorted(set(np.abs(raw_ks))))
        k_axis = k_box

    p_spec = []
    p_spec_err = []

    for i in range(len(k_axis)):
        if i%(len(k_axis)//10) == 0:
            print("Starting k number %d out of %d" % (i, len(k_axis)))
        k = k_axis[i]
        locs = (k_box == k).nonzero()
        pwr = np.mean(f_dist[locs]*np.conj(f_dist[locs]))
        p_spec.append(pwr)
        p_spec_err.append(pwr/np.sqrt(len(locs[0])))

    p_spec = np.array(np.real(p_spec))
    # p_spec[0] = 0.0
    p_spec_err = np.array(p_spec_err)

    return k_axis, p_spec, p_spec_err


def make_fourier_space(p_spec,shape,dx):
    dims = len(shape)
    center = shape//2
    fs = []
    for n in range(dims):
        fs.append(np.fft.fftshift(np.fft.fftfreq(shape[n]))/dx)
    fs = np.array(fs)
    raw_ks = fs*2*np.pi

    #Get box with the shape of the given dist with the k-value of each cell
    if dims != 1:
        raw_k_box = np.array(np.meshgrid(*raw_ks))
        if dims == 2:
            raw_k_box = np.einsum('kji->ijk',raw_k_box)
        elif dims == 3:
            raw_k_box = np.einsum('ljik->ijkl',raw_k_box)
        k_box = np.sqrt(np.sum(raw_k_box**2,axis=dims)) #Box of the wavenumber of every position at that position
        k_axis = np.array(sorted(set(np.abs(k_box).flatten()))) #List of unique k-values in fourier space
    else:
        k_box = np.array(sorted(set(np.abs(raw_ks))))
        k_axis = k_box

    four_dist = np.zeros(shape,dtype=complex)
    btm_half_ish = np.zeros(shape)-1
    if dims != 1:
        btm_half_ish[:shape[0]//2] = k_box[:shape[0]//2]
        btm_half_ish[shape[0]//2,:shape[1]//2] = k_box[shape[0]//2,:shape[1]//2]
        if dims == 3:
            btm_half_ish[shape[0]//2,shape[1]//2,:shape[2]//2] = k_box[shape[0]//2,shape[1]//2,:shape[2]//2]
        else:
            print("Can't handle that many dimensions yet, sorry.")
            quit()
    else:
        btm_half_ish[:shape//2+1]

    for i in range(1,len(k_axis)):
        if i%(len(k_axis)//10) == 0:
            print("Starting k number %d out of %d" % (i, len(k_axis)))
        k = k_axis[i]
        locs = (btm_half_ish == k).nonzero()
        num = len(locs[0])
        # end = np.array([shape]*num).T - 1
        offset = np.array([shape%2]*num).T
        neg_locs = tuple(-(locs+offset).astype(int))
        std_dev_sqrd = p_spec[i]
        
        re_vals = np.random.normal(loc=0, scale=np.sqrt(std_dev_sqrd/2.0), size=num)
        im_vals = 1j*np.random.normal(loc=0, scale=np.sqrt(std_dev_sqrd/2.0), size=num)
        entries = re_vals + im_vals

        four_dist[locs] = entries
        four_dist[neg_locs] = np.conj(entries) #To ensure fourier distribution that would produce real data when inverse fourier transformed

    four_dist[shape[0]//2,shape[1]//2,shape[2]//2] = np.random.normal(loc=0.0, scale=np.sqrt(p_spec[0]/2.0), size=1)

    return np.fft.ifftshift(four_dist)



def main():

    ##################################################################
    # Initialize Space and Distribution Corresponding to 3D sin Wave #
    ##################################################################
    
    L = 100
    dx = 0.5
    k = np.pi
    # test_dist = np.random.normal(loc = 1.0, scale = 10,size=(L,L,L))
    x = np.arange(0,100*dx,dx)
    x_box = np.array(np.meshgrid(x,x,x))
    print(x_box.shape)
    x_box = np.einsum('ljik->ijkl',x_box)
    x_box = x_box - np.array([L*dx//2,L*dx//2,L*dx//2])
    x_dist = np.sqrt(np.sum((x_box)**2,axis=3))
    test_dist = np.sin(k*x_dist)
    plt.imshow(test_dist[len(test_dist)//2,:,:])
    plt.colorbar()
    plt.title("Slice of Initial Distribution")
    plt.axis('off')
    plt.savefig("init_test_dist.png")
    plt.show()
    plt.clf()
    
    test_dist_ft = np.fft.fftshift(np.fft.fftn(test_dist,norm="ortho"))
    plt.imshow(np.real(test_dist_ft[len(test_dist_ft)//2]))
    plt.colorbar()
    plt.title("Fourier Transform of Initial Distribution (Slice)")
    plt.axis('off')
    plt.savefig("ft_init_test_dist.png")
    plt.show()
    plt.clf()

    # test_dist_ift = np.fft.ifftn(test_dist_ft,norm="ortho")
    # plt.imshow(np.real(test_dist_ift[len(test_dist_ift)//2,:,:]))
    # plt.show()
    # plt.clf()

    
    ##########################################################################
    # Get power Spectrum from Distribution to See if it has the correct form #
    # and if I know anything about how to take a power spectrum              #
    ##########################################################################
    
    k_axis, p_spec, p_spec_err = get_p_spec(test_dist_ft, dx)
    plt.errorbar(k_axis,np.real(p_spec),yerr=p_spec_err,ecolor='r')
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.title("Power Spectrum Test")
    plt.axvline(x=np.pi,color='k')
    plt.savefig("p_spec_init_test_dist.png")
    plt.show()
    plt.clf()

    p_spec_file = open("test_p_spec.txt","w")
    p_spec_file.write(str(p_spec.tolist()))
    p_spec_file.close()

    #For when I just want to retrieve the p_spec from a file instead of going through the whole process again
    # p_spec_str = open("test_p_spec.txt","r").read()
    # p_spec = np.fromstring(p_spec_str.strip("[").strip("]"),sep=", ", dtype=np.float64)
    # plt.plot(np.arange(len(p_spec)),p_spec)
    # plt.show()
    # plt.clf()

    
    ##########################################################
    # Put power spectrum into fctn and produce fourier space #
    # that hopefully corresponds to a distribution           #
    # that roughly respects the initial power spectrum       #
    ##########################################################

    shape = np.array([L,L,L])
    four_dist = get_four_dist(p_spec,shape,dx) #Function returns fourier distribution with dc term at 0 point so that it is numerically ready to just be inverse transformed.
    plt.imshow(np.real(np.fft.fftshift(four_dist[shape[0]//2,:,:]))) #Shift before plotting so that dc term is in center, to better fit our brains visual intuition.
    plt.colorbar()
    plt.title("Slice of New Fourier Space")
    plt.axis('off')
    plt.savefig("ft_new_dist.png")
    plt.show()
    plt.clf()

    #Got Fourier distribution, now let's see what it looks like in space-space
    new_dist = np.real(np.fft.ifftn(four_dist,norm="ortho"))
    plt.imshow(new_dist[shape[0]//2,:,:])
    plt.colorbar()
    plt.title("Slice of New Distribution")
    plt.axis('off')
    plt.savefig("new_dist.png")
    plt.show()
    plt.clf()

    
    ########################################################################
    # See if new distribution resulting from produced Fourier distribution #
    # actually has a power spectrum that resembles the initial one         #
    # we were trying to immitate                                           #
    ########################################################################    

    new_dist_ft = np.fft.fftshift(np.fft.fftn(new_dist,norm='ortho'))
    k_axis,new_p_spec,new_p_spec_errs = get_p_spec(new_dist_ft,dx)
    plt.errorbar(k_axis, np.real(new_p_spec), yerr=new_p_spec_errs, ecolor='r')
    plt.title("Power Spectrum of New Distribution")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.axvline(x=k,color='k')
    plt.savefig("p_spec_new_dist.png")
    plt.show()



if __name__ == "__main__":
    main()

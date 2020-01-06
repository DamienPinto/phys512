import numpy as np
import math
import os
import sys
import numpy.random
import matplotlib.pyplot as plt
from p_spec_tools import make_power_spectrum
from get_array_from_txt import get_array_from_txt_w_list


#Receive a power spectrum as an array where each value is the std_dev squared of the Gaussian noise distribution
#that produced the power spectrum but also the value of the Fourier transform at that value's index. The index is
# k = np.sqrt(k1**2 + k2**2 + k3**2) and is contributed to by any combination of values if k1, k2 #& k3 who, when 
#combined in a vector (k1, k2, k3) have a norm of k. Want to take that power spectrum and produce a Fourier Space 
#that has a power spectrum similar in statistical characteristics to the one entered. Has to respect F(-k) = [F(k)]* 
#to ensure real output when #translating to actual matter distribution. I'm only hoping to make cubic Fourier Spaces here.

DATA_DIR = "Data Files/"



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




# def make_fourier_space(power_spectrum, dim):

# 	#For now just making a cubic region, can't assume shape of Fourier space used to generate power spectrum without errors.
# 	side_length = int(round((len(power_spectrum)/np.sqrt(dim))))
# 	# print(side_length)
# 	# print(np.array(power_spectrum).shape)

# 	#I know we discussed that spaces we use should have dimensions that are powers of two, however this method was used 
# 	#to ensure that all k values can be used and that a vaue can also be placed at -k.
# 	#So, for example, for k = 3, we need a space to cell to insert a value at k1 = 3 but also k1 = -3.
# 	#So I figured twice the k_max - 1 (because the origin will fold on itself).
# 	if dim == 3:
# 		fourier_universe = np.zeros((2*side_length, 2*side_length, 2*side_length), dtype = complex)
# 	elif dim == 2:
# 		fourier_universe = np.zeros((2*side_length, 2*side_length), dtype = complex)
# 	#Array containing arrays, each of which corresponds to a specific integer value of k.
# 	shell_register   = [[] for _ in range(len(power_spectrum))]

# 	#Go through all the coordinates, determine their integer distance from the origin and append that set of coordinates
# 	#to the shell_register in the corresponding array.
# 	for k1 in range(-side_length+1, side_length):
# 		for k2 in range(-side_length+1, side_length):
# 			if len(fourier_universe.shape) == 3:
# 				for k3 in range(side_length):
# 					shell_register[int(round((np.sqrt(k1**2+k2**2+k3**2))))].append((k1, k2, k3))
# 			elif len(fourier_universe.shape) == 2:
# 				shell_register[int(round((np.sqrt(k1**2+k2**2))))].append((k1, k2))


# 	for i in range(len(shell_register)):

# 		#Get list of coordinates with same value for their norm from the origin as well as the power spectrum value for 
# 		#that shell.
# 		norm_val_list = shell_register[i]
# 		std_dev_sqrd  = power_spectrum[shell_register.index(shell_register[i])]

# 		#Dr.Adrian's method of centering distributions around 0 and giving them the same standard deviation as the value
# 		#of the power spectrum.
# 		re_vals = np.random.normal(loc=0, scale=np.sqrt(std_dev_sqrd/2.0), size=len(norm_val_list))
# 		im_vals = np.random.normal(loc=0, scale=np.sqrt(std_dev_sqrd/2.0), size=len(norm_val_list))


# 		#My method using a noral distribution to chose values in Fourier space
# 		# vals = np.random.normal(loc = std_dev_sqrd, scale=10, size=len(norm_val_list))

# 		#My method but just randomly chosing positive values to place in Fourier space
# 		# while not(all(val > 0 for val in vals)):

# 		# 	vals = np.random.normal(loc = std_dev_sqrd, scale=10, size=len(norm_val_list))
# 		for j in range(len(norm_val_list)):

# 			enterred_val = np.complex(re_vals[j], im_vals[j])

# 			# partitioning = np.random.rand()

# 			# enterred_val = np.complex(real_sign*np.sqrt(partitioning*vals[i]), complex_sign*np.sqrt((1-partitioning)*vals[i]))


# 			#We thought of shifting here, but after experimenting with the fast fourier transform (fft) and inverse fast
# 			#fourier transform (ifft) functions I saw that the output of fft and the input that ifft takes before the 
# 			#shifts is one where the first value is the origin, the next entry being the value for the next smallest 
# 			#frequecy in Fourier, then the next... once it reaches the largest grequecy in the Fourier space it cycles 
# 			#back to the largest negative frequency, so kind of [0, f1, f2, ... fN, -fN, -fN-1, ... -f1] so if I saw 
# 			#that correctly then this should work.

# 			if len(fourier_universe.shape) == 3:
# 				fourier_universe[norm_val_list[j][0]][norm_val_list[j][1]][norm_val_list[j][2]]	= enterred_val

# 				fourier_universe[-norm_val_list[j][0]][-norm_val_list[j][1]][-norm_val_list[j][2]] = np.conj(enterred_val)
# 			elif len(fourier_universe.shape) == 2:
# 				fourier_universe[norm_val_list[j][0]][norm_val_list[j][1]]	= enterred_val

# 				fourier_universe[-norm_val_list[j][0]][-norm_val_list[j][1]] = np.conj(enterred_val)

# 		if len(fourier_universe.shape) == 3:
# 			fourier_universe[0,0,0] = np.real(fourier_universe[0,0,0])
# 		elif len(fourier_universe.shape) == 2:
# 			fourier_universe[0,0] = np.real(fourier_universe[0,0])

# 	return fourier_universe

 
	

def main():
	p_spec_filename = sys.argv[1]
	errs_filename = sys.argv[2]
	dim = int(sys.argv[3])

	if os.path.isfile(DATA_DIR + p_spec_filename) and os.path.isfile(DATA_DIR + errs_filename):
		print("Retrieving Power Spectrum and Errors")
		p_spec = get_array_from_txt_w_list(DATA_DIR + p_spec_filename)
		p_spec_errs = get_array_from_txt_w_list(DATA_DIR + errs_filename)

		distribution = get_dist_from_p_spec(p_spec, dim)
		OUT_PATH = "Output Files/dist_from_p_spec.txt"
		print("Obtained distribution of dimensions " + str(np.shape(distribution)) + " from Fourier Space. Printing flattened version to file: " + OUT_PATH)
		
		output_file = open(OUT_PATH, "w")
		output_file.write(str(distribution.tolist()))
		output_file.close()

	else:
		print("One or both of your input files could not be found. Please verify their spelling and the location specified by the DATA_DIR variable and try again.")
		quit()

	return



if __name__ == '__main__':
	main()

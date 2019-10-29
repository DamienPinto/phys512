import numpy as np
import camb
from matplotlib import pyplot as plt
import time

#### Prof. Sievers' Part ###############################################################################################

def get_spectrum(pars,lmax=2000):
    #print('pars are ',pars)
    H0=pars[0]    #Hubble Constant
    ombh2=pars[1] #Physical Baryon Density
    omch2=pars[2] #Cold Dark Matter Density
    As=pars[3]    #Primordial Amplitude of Fluctuations
    ns=pars[4]    #Slope of the Primordial Power Law
    tau=pars[5]   #Optical Depth
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt


#plt.ion()

pars=np.asarray([65,0.02,0.1,2e-9,0.96,0.05])
wmap=np.loadtxt('../wmap_tt_spectrum_9yr_v5.txt')
# print("Length of wmap[:,0]: ", len(wmap[:, 0]))

plt.clf();
plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*')
#plt.plot(wmap[:,0],wmap[:,1],'.')
#plt.show()
l          = len(wmap[:,1])
# before1199 = time.clock()
cmb        = get_spectrum(pars, lmax=l)[2:]
# after1199  = time.clock()
# time1199   = after1199 - before1199
# print("Time for 1199 points is: ", time1199)
plt.plot(cmb)
plt.show()

# N = 500
# before500 = time.clock()
# cmb2      = get_spectrum(pars, lmax=N)[2:]
# print("Length of cmb2: ", len(cmb2))
# after500  = time.clock()
# time500   = after500 - before500
# print("Time for %d is: %lf" % (N, time500))
# plt.clf()
# plt.plot(wmap[:,0], cmb)
# print("(len(wmap[:,0])//N): ", (len(wmap[:,0])//N))
# red_wmap0 = wmap[::(len(wmap[:,0])//N),0]
# print(red_wmap0)
# print("Length of red_wmap0: ", len(red_wmap0))
# plt.plot(red_wmap0, cmb2)
# idx = np.round(np.linspace(0, len(wmap[:,0])-1, len(cmb2))).astype(int)
# x = (wmap[:,0])[idx]
# plt.plot(x, cmb2)
# plt.plot(wmap[:,0], cmb)
# plt.show()



#### Prof. Sievers' Part (END) #########################################################################################
r    = wmap[:,1]-cmb
N    = wmap[:,2]
chi2 = np.sum((r/N)**2)

print("Chi^2 is : %f" % chi2)
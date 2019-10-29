import numpy as np
import matplotlib.pyplot as plt
import camb




def get_spectrum(pars,lmax=1199):
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
    return tt[2:]




def get_ders(params, p_spec_og):
	dbl_pres = 1e-12

	dx_lin = np.linspace(0.0001, 1, 100)

	for i in range(len(params)):
		for dx in dx_lin:
			dp 		   = np.zeros(len(params))
			dp[i] 	   = dp[i]*(1+dx)
			params_pdp = params + 2*dp
			params_mdp = params - 2*dp
			print("Getting p_spec_pdp")
			p_spec_pdp = get_spectrum(params_pdp)
			print("Getting p_spec_mdp")
			p_spec_mdp = get_spectrum(params_mdp)
			print("Done")

			# plt.clf()
			func = np.sqrt(np.sqrt(2.0*dbl_pres*p_spec_og/(p_spec_pdp - p_spec_mdp)))
			print(func.max())
			# xx = np.arange(len(func))
			# plt.hlines(dx, xx[0], xx[-1])
			# plt.show()

			




pars=np.asarray([65,0.02,0.1,2e-9,0.96,0.05])
wmap=np.loadtxt('../wmap_tt_spectrum_9yr_v5.txt')

p_spec_og = get_spectrum(pars)

get_ders(pars, p_spec_og)
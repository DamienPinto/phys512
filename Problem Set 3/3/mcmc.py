import numpy as np
import matplotlib.pyplot as plt
import camb
import time
from get_array_from_txt import get_array_from_txt_w_list




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




def get_stats_from_chains(chains):
	avg_params = np.mean(chains, axis=0)
	std_params = np.std(chains, axis=0)

	return avg_params, std_params




def get_rdm_step(scale_factor=1.0):
	param_scalings = np.asarray([0.1, 1e-3, 1e-2, 1e-11, 1e-2, 1e-2])
	dp 	  =  scale_factor*param_scalings*np.random.randn(len(param_scalings))

	return dp



#Prof. Sievers' code (just renamed some stuff to be more intuitive for me):
def take_cov_step(cov_mat, scl_fctr):
	chol_mat = np.linalg.cholesky(cov_mat)
	noisy_chol_mat = scl_fctr*np.squeeze(np.asarray((chol_mat @ np.random.randn(cov_mat.shape[0]))))

	return noisy_chol_mat




def tau_prior(tau):
	mu_tau  = 0.0544
	sig_tau = 0.0073

	return np.exp(-0.5*(tau - mu_tau)**2/sig_tau**2)




def mcmc(y, params, pred_func, scl_fctr=0.5, tau_fctr=0.1, frac_burn=0.1, max_iter=500, gauss_N=False, cov_informed=False, cov_mat=[], filename="mcmc_chain_"+str(round(time.perf_counter()))+".txt"):
	
	if gauss_N:
		N = y[1]
		y = y[0]

		N = np.diag(N)**2
	else:
		N = np.ones(len(y))
		N = np.diag(N)

	file = open(filename, "w")
	file.write("H0, ombh2, omch2, As, ns, tau, chi2\n")
	file.write("[")
	#Huble cst, phys. Baryon density, cold dark matter density, 
	#Primordial amplitude offluctuations, slope of primordial power law
	#optical depth, chi squared

	#setup
	num_burn = int(max_iter*frac_burn)
	chains 	 = np.zeros([num_burn + max_iter, len(params)])
	chi2_log = np.zeros(num_burn + max_iter)
	num_acc  = 0

	#If you somehow know how many steps you want to/should burn, then you can use this instead of the previous similar
	#lines. 
	'''
	chains 	 = np.zeros([max_iter, len(params)])
	chi2_log = np.zeros(max_iter)
	'''

	#setup but do/compute some stuff
	pred = pred_func(params)
	r 	 = y - pred
	chi2 = (r.T @ np.linalg.inv(N)) @ r

	for i in range(-num_burn, max_iter):
		print("MCMC at step %d, accepted %d " % (i, num_acc))
		if cov_informed:
			dp = take_cov_step(cov_mat, scl_fctr)
		else:
			# print("mcmc: here 1")
			dp = get_rdm_step(scl_fctr)
		dp[-1] = tau_fctr*dp[-1]
		params_tmp = params + dp
		# print("mcmc: here 2")
		print(params_tmp)
		# print(params_tmp[-1])
		if params_tmp[-1] >= 0: 
			# print("mcmc: here 2.1")
			params_tmp = params + dp
			# print("mcmc: here 3")
			# print(params_tmp)
			pred_tmp   = pred_func(params_tmp)
			# print("mcmc: here 4")
			r_tmp 	   = y - pred_tmp
			# print("mcmc: here 5")
			chi2_tmp   = (r_tmp.T @ np.linalg.inv(N)) @ r_tmp
			# print("mcmc: here 6")


			delta_chisq = chi2_tmp - chi2 		  ##### tau restriction #####
			prob 		= np.exp(-0.5*delta_chisq)#*tau_prior(params_tmp[-1])/0.6065
			print("chi2_tmp: ", chi2_tmp)
			print("chi2", chi2)
			print("delta_chisq: ", delta_chisq)
			print("prob: ", prob)

			if np.random.rand(1) < prob:
				params  = params_tmp
				pred    = pred_tmp
				r 	    = r_tmp
				chi2    = chi2_tmp
				num_acc += 1
		else:
			print("Negative Tau, not accepting step.")

		#You want to keep trakc of these even if the steps weren't accepted to see when/if your walkers stagnate in a
		#specific area of parameter space (which will probably be a local/global solution(?))
		entry = str(list(np.concatenate((params, [chi2])))).replace("  ", ", ") + ", "
		if i == max_iter-1:
			entry = entry[:-2] + "]"
		file.write(entry)
		file.flush()
		# print("mcmc: here 7")
		chains[i+num_burn,:] = params
		chi2_log[i+num_burn] = chi2

		#if you know beforehand how many steps you wan to burn and don't want to see/log them:
		'''
		if i >= 0:
			chains[i,:] = params
			chi2_log[i] = chi2
		'''
	file.close()
	print("%lf percent of steps accepted." % (num_acc/i))
	return chains, chi2_log, N




def main():
	pars  = np.asarray([65,0.02,0.1,2e-9,0.96,0.05])
	guess = get_spectrum(pars)
	wmap  = np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

	scl_fctr = 0.4
	tau_fctr = 0.05
	# print("Here main 1")
	# chains, chi2_log, N = mcmc([wmap[:,1], wmap[:,2]], pars, get_spectrum, scl_fctr=scl_fctr, tau_fctr=tau_fctr, max_iter=500, gauss_N=True, cov_informed = False, filename="mcmc_chain_simple2.txt")
	# print("Here main 2")
	# avg_params, std_params = get_stats_from_chains(chains)
	# print("avg_params: ", avg_params)
	# print("std_params: ", std_params)
	# mcmc_fit  = get_spectrum(avg_params)
	# r 		  = wmap[:, 1] - mcmc_fit
	# mcmc_chi2 = (r.T @ np.linalg.inv(N)) @ r
	# print("Here main 3")

	# print("Simple MCMC fit done.")
	# print("Best params found are: ", avg_params)
	# print("Error on Params: ", std_params)

	# fig, axes = plt.subplots(2, 1, sharex=True, dpi=255)

	# axes[0].set_title("MCMC Fit Results")
	# axes[0].set_ylabel("Power Spectrum of CMB")
	# axes[0].errorbar(wmap[:,0], wmap[:,1], wmap[:,2], fmt=".", label="CMB data", markersize=1, elinewidth=0.5)
	# axes[0].plot(wmap[:,0], guess, linestyle="--", label="First Guess", linewidth=0.7)
	# axes[0].plot(wmap[:,0], mcmc_fit, linestyle="-", markersize=5, linewidth=0.7)

	# axes[1].hlines(0, 0.5, 1)
	# axes[1].set_ylabel("MCMC Residuals")
	# axes[1].set_xlabel("k")
	# axes[1].scatter(wmap[:,0], r, s=0.5)
	# axes[1].set_ylim([min(r)*1.1, max(r)*1.1])

	# plt.savefig("MCMC_simple_fit.png")
	# plt.show()
	# plt.clf()

	# simple_fit_filename = "simple_fit_data.txt"
	# simple_fit_file 	= open(simple_fit_filename, "w")
	# simple_fit_file.write(str(list(mcmc_fit)))
	# simple_fit_file.close()

	# simple_fit_params_filename = "simple_fit_params_n_err.txt"
	# simple_fit_params_file 	   = open(simple_fit_params_filename, "w")
	# simple_fit_params_file.write(str(list(avg_params)))
	# simple_fit_params_file.write("\n")
	# simple_fit_params_file.write(str(list(std_params)))
	# simple_fit_params_file.close()

	cov_file = "final_fit_with_tau_flat_cov_mat.txt"
	cov_mat  = get_array_from_txt_w_list(cov_file).reshape((len(pars), len(pars)))

	print("Starting MCMC Fit Informed by Covariance Matrix of Levenberg-Marquardt Fit.")
	chains_cov, chi2_log_cov, N = mcmc([wmap[:,1], wmap[:,2]], pars, get_spectrum, scl_fctr=scl_fctr, tau_fctr=tau_fctr, frac_burn=0.0, max_iter=10000, gauss_N=True, cov_informed=True, cov_mat=cov_mat, filename="10000_mcmc_chain_cov_restr_tau" + str(tau_fctr) + "_scl" + str(scl_fctr) + ".txt")

	avg_params_cov, std_params_cov = get_stats_from_chains(chains_cov)
	mcmc_fit_cov 	 			   = get_spectrum(avg_params_cov)
	r_cov 		 				   = wmap[:, 1] - mcmc_fit_cov
	mcmc_chi2_cov 				   = (r_cov.T @ np.linalg.inv(N)) @ r_cov

	print("Covariance-Informed MCMC fit done.")
	print("Best params found are: ", avg_params_cov)
	print("Error on Params: ", std_params_cov)

	fig_cov, axes_cov = plt.subplots(2, 1, sharex=True, dpi=255)

	axes_cov[0].set_title("MCMC Covariance-Informed Fit Results")
	axes_cov[0].set_ylabel("Power Spectrum of CMB")
	axes_cov[0].errorbar(wmap[:,0], wmap[:,1], wmap[:,2], fmt=".", label="CMB data", markersize=1, elinewidth=0.5)
	axes_cov[0].plot(wmap[:,0], guess, linestyle="--", label="First Guess", linewidth=0.7)
	axes_cov[0].plot(wmap[:,0], mcmc_fit_cov, linestyle="-", markersize=5, linewidth=0.7)

	axes_cov[1].hlines(0, 0.5, 1)
	axes_cov[1].set_ylabel("MCMC Residuals")
	axes_cov[1].set_xlabel("k")
	axes_cov[1].scatter(wmap[:,0], r_cov, s=0.5)
	axes_cov[1].set_ylim([min(r_cov)*1.1, max(r_cov)*1.1])

	plt.savefig("10000_MCMC_cov_fit_restr_tau" + str(tau_fctr) + "_scl" + str(scl_fctr) + ".png")
	plt.show()

	cov_fit_filename = "10000_cov_fit_data_restr_tau" + str(tau_fctr) + "_scl" + str(scl_fctr) + ".txt"
	cov_fit_file 	 = open(cov_fit_filename, "w")
	cov_fit_file.write(str(list(mcmc_fit_cov)))
	cov_fit_file.close()

	cov_fit_params_filename = "10000_cov_fit_params_n_err_restr_tau" + str(tau_fctr) + "_scl" + str(scl_fctr) +".txt"
	cov_fit_params_file 	= open(cov_fit_params_filename, "w")
	cov_fit_params_file.write(str(list(avg_params_cov)))
	cov_fit_params_file.write("\n")
	cov_fit_params_file.write(str(list(std_params_cov)))
	cov_fit_params_file.close()



if __name__ == "__main__":
	main()
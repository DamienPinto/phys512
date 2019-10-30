import numpy as np
import matplotlib.pyplot as plt
import camb
import corner




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




def get_mcmc_data(filename):
	
	mcmc_file 	 = open(filename, "r")
	mcmc_str  	 = mcmc_file.read()
	mcmc_str  	 = mcmc_str[37:]
	mcmc_str_arr = mcmc_str.split("], [")
	mcmc_arr 	 = np.array([np.fromstring(step.strip("[").strip("]"), sep=", ") for step in mcmc_str_arr])
	print(mcmc_arr.shape)

	return mcmc_arr




def main():
	wmap 	 = np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
	filename = "10000_mcmc_chain_cov_restr_tau0.05_scl0.4.txt"
	chains1 	 = get_mcmc_data(filename)

	##### Corner Plot Code ######
	# chains[:, 3] = chains[:, 3]*1e9
	# titles=[r"$H_0$", r"$\Omega_bh^2$", r"$\Omega_ch^2$", r"$A_s$", r"$n_s$", r"$\tau$"]
	# titles = ["H0", "Ombh2", "Omch2", "As[1e-9]", "ns", "tau", "chi2"]
	# plt_corner_fig, plt_corner_axes = plt.subplots(7,7)
	# plt_corner_fig.axes = [7,7]
	# corner_fig = corner.corner(chains, show_titles=True, labels=titles, plot_datapoints=True, fig=plt_corner_fig)
	# plt.show(plt_corner_fig)
	# plt.savefig("corner_cov_mcmc.png")
	# print("passed corner")
	# quit()
	##### Corner Plot Code (END) #####

	##### Planck Compatible Tau Fit #####
	# filename2 = "MCMC Simple/mcmc_chain_simple2.txt"
	# chains2 = get_mcmc_data(filename2)
	# planck_tau = 0.0544
	# planck_std = 0.0073
	# chains1_tau_slct = chains1[np.where(np.logical_and(chains1[:,-2]>=(planck_tau - planck_std), chains1[:,-2]<=(planck_tau + planck_std)))]
	# chains2_tau_slct = chains2[np.where(np.logical_and(chains2[:,-2]>=(planck_tau - planck_std), chains2[:,-2]<=(planck_tau + planck_std)))]

	# print("Length of chains1_tau_slct is %d and length of chains2_tau_slct is %d." % (len(chains1_tau_slct), len(chains2_tau_slct)))
	# chains_comb = np.concatenate((chains1, chains2))
	# avg_params_tau_slct = np.mean(chains_comb, axis=0)
	# std_params_tau_slct = np.std(chains_comb, axis=0)

	# print("Avg Planck Compatible Params: ", avg_params_tau_slct)
	# print("Std_dev Planck Compatible Params: ", std_params_tau_slct)

	# tau_slct_fit  = get_spectrum(avg_params_tau_slct[:-1], lmax=len(wmap[:,0]))
	# tau_slct_pdp  = get_spectrum((avg_params_tau_slct[:-1] + np.abs(std_params_tau_slct)[:-1]), lmax=len(wmap[:,0]))
	# tau_slct_mdp  = get_spectrum((avg_params_tau_slct[:-1] - np.abs(std_params_tau_slct)[:-1]), lmax=len(wmap[:,0]))
	# r_tau_slct    = wmap[:, 1] - tau_slct_fit
	# N 			  = np.diag(wmap[:,2])
	# N2 			  = N**2
	# chi2_tau_slct = r_tau_slct.T @ np.linalg.inv(N2) @ r_tau_slct
	# print("Chi2 of Planck compatible fit: ", chi2_tau_slct)

	# plt.clf()

	# fig_tau_slct, axes_tau_slct = plt.subplots(2,1, sharex=True, dpi=255)

	# axes_tau_slct[0].set_title(r"MCMC Restricted $\tau$ Fit Results")
	# axes_tau_slct[0].set_ylabel("[Insert Correct Units Here]")
	# axes_tau_slct[0].errorbar(wmap[:,0], wmap[:,1], wmap[:,2], fmt=".", markersize=1, elinewidth=0.5, label="WMAP Data", alpha=0.3)
	# axes_tau_slct[0].plot(wmap[:,0], tau_slct_fit, label=r"MCMC Average $\tau$ Selection Fit")
	# axes_tau_slct[0].fill_between(wmap[:,0], tau_slct_pdp, tau_slct_mdp, color='grey', alpha=0.5, label=r"$1\sigma$ Parameter Variation")
	# axes_tau_slct[0].legend(loc="lower left", fontsize=5)
	
	# axes_tau_slct[1].hlines(0, 0.5, 1)
	# axes_tau_slct[1].set_xlabel(r"$k[\lambda \cdot Mpc^{-1}]$")
	# axes_tau_slct[1].set_ylabel("Residuals")
	# axes_tau_slct[1].scatter(wmap[:,0], r_tau_slct, s=0.5)
	# axes_tau_slct[1].set_ylim([min(r_tau_slct)*1.1, max(r_tau_slct)*1.1])

	# plt.savefig("planck_compatible_fit.png")
	# plt.show()

	# quit()
	##### Planck Compatible Tau Fit (END) #####


	avg_params, std_params = get_stats_from_chains(chains)
	print("Avg params: ", avg_params)
	print("Std params: ", std_params)

	mcmc_fit 	 = get_spectrum(avg_params[:-1], lmax=len(wmap[:,0]))
	mcmc_fit_pdp = get_spectrum((avg_params + np.abs(std_params))[:-1], lmax=len(wmap[:,0]))
	mcmc_fit_mdp = get_spectrum((avg_params - np.abs(std_params))[:-1], lmax=len(wmap[:,0]))

	r = wmap[:,1] - mcmc_fit
	N = np.diag(wmap[:,2])
	N2 = N**2
	chi2 = r.T @ np.linalg.inv(N2) @ r
	print("Chi2 is: ", chi2)

	fig, axes = plt.subplots(2, 1, sharex=True, dpi=255)

	axes[0].set_title("MCMC Random Fit Results")
	axes[0].set_ylabel("[Insert Correct Units Here]")
	axes[0].errorbar(wmap[:,0], wmap[:,1], wmap[:,2], fmt=".", markersize=1, elinewidth=0.5, label="WMAP Data", alpha=0.3)
	axes[0].plot(wmap[:,0], mcmc_fit, label="MCMC Average Fit")
	axes[0].fill_between(wmap[:,0], mcmc_fit_pdp, mcmc_fit_mdp, color='grey', alpha=0.5, label=r"$1\sigma$ Parameter Variation")
	axes[0].legend(loc="lower left", fontsize=5)
	
	axes[1].hlines(0, 0.5, 1)
	axes[1].set_xlabel(r"$k[\lambda \cdot Mpc^{-1}]$")
	axes[1].set_ylabel("Residuals")
	axes[1].scatter(wmap[:,0], r, s=0.5)
	axes[1].set_ylim([min(r)*1.1, max(r)*1.1])

	plt.savefig("550step_mcmcRandom_best_fit.png")
	plt.show()






if __name__ == "__main__":
	main()
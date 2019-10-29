import numpy as np
from matplotlib import pyplot as plt
import camb
import time

'''
What I want to do:
	-Write a Levenberg-Marquardt fitter

What does that require/entail:
	-Getting matrix of derivatives (A) of p_spec wrt each variable -> evaluate p_spec for m_i +/- dm_i and do
	 double-sided derivative
	-Implement dm = (A'.T*N^{-1}*A' + (lambda)*I)*(A.T*N^{-1}*r)
						^approx curvature	  ^simple grad. descent
	-Levenberg-Marquardt: start lambda small-ish, if chi^2 goes down after sted, lambda->lambda/cst, else lambda->cst*lambda
'''




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




def get_spectrum_hold_tau(params):
	TAU    = 0.05
	params = np.concatenate((params, [TAU]))
	p_spec = get_spectrum(params)

	return p_spec




def get_cdtn_num(A):
	A 		 = np.matrix(A)
	mtx2 	 = A.T*A
	mtx2_sym = mtx2 + mtx2.T
	es,vecs  = np.linalg.eig(mtx2_sym)
	eabs 	 = np.abs(es)
	emin = eabs.min()
	if eabs.min() == 0.0:
		print("eabs min was 0, setting to one.")
		emin = 1
	cond 	 = eabs.max()/emin

	return cond




def get_Ap(func, params, scl=100, v=False):

	#Arbitrary fraction of 1/200th, just seemed to be a good scale for this problem for the little "wiggle" we do
	#in parameter space to get the gradient of where we're at. Not sure what a quantitative way of determining the size
	#of the wiggle to take as a function of the range in magnitude of the parameters provided.
	dp = params/scl
	
	if np.array(params).shape == ():
		sgl = True
	else:
		sgl = False

	if v:
		print("Starting construction of Ap...")

	if sgl:
		params_mat = params
		dp_diag    = dp
	else:
		params_mat = np.array([params]*len(params))
		dp_diag	   = np.diag(dp)

	#make matrix where each row is our parameters with one incremented positively with its dp:
	params_pdp = params_mat + dp_diag
	#make matrix where each row is our parameters with one incremented negatively with its dp:
	params_mdp = params_mat - dp_diag

	if sgl:
		data_pdp = np.array(func(params_pdp))
		data_mdp = np.array(func(params_mdp))
		dp_rep   = dp
	else:
		if v:
			print("...made arrays of different parameter arrangements...")
			print("...getting data for params +/- dp...")
		data_pdp = np.array([func(params_pdp[i]) for i in range(len(params_pdp))])
		data_mdp = np.array([func(params_mdp[i]) for i in range(len(params_mdp))])
		if v:
			print("...obtained data for params +/- dp...")
			print("...computing derivatives at all points...")
		dp_rep = np.array([[dp_i]*len(data_pdp[0]) for dp_i in dp])

	Ap = ((data_pdp - data_mdp)/(2*dp_rep)).T #A' = Ap
	if v:
		print("..done. Returning Ap.")

	cdtn = get_cdtn_num(Ap)
	# print("Condition number of Ap is: ", cdtn)

	return Ap, cdtn




def lev_marq(x, y, params, pred_func, Ap_func, lamb_init, chi_tol=0.01, max_iter=100, gauss_N=False):
	#If the sigma for gaussian noise are provided, use that to construct N,
	if gauss_N:
		N 	   = y[1]
		y 	   = y[0]
		N_diag = np.diag(N)**2
	#else, just make N the identity matrix
	else:
		N 	   = np.ones(len(y))
		N_diag = np.diag(N)

	#For functions that retrieve A but aren't based on analytical derivatives, 
	#need to generate another data set with perturbed params to generate A.
	Ap, cdtn, scl = Ap_func(pred_func, params, v=True)

	pred  = pred_func(params)
	r 	  = np.array(y - pred)
	chi2  = np.dot(np.dot(r, np.linalg.inv(N_diag)), r.T)

	lamb 	 = lamb_init
	chi_ctr  = 0
	lamb_ctr = 0
	cdtn_array = np.zeros(len(max_iter))
	for i in range(max_iter):
		cdtn_array[i] = cdtn
		#curvature = curv = A'.T*N^{-1}*A'
		#dp = (curv + (lambda)*diag(curv))^{-1}*(A'.T*N^{-1}*r)
		curv = np.dot(np.dot(Ap.T, np.linalg.inv(N_diag)), Ap)
		grad = np.diag(np.diag(curv))
		lhs  = np.linalg.inv(curv + lamb*grad)
		rhs  = np.dot(np.dot(Ap.T, np.linalg.inv(N_diag)), r)
		dp 	 = np.dot(lhs,rhs).flatten()

		params_tmp = params+dp
		pred_tmp   = pred_func(params_tmp)
		pred_tmp   = pred_tmp
		r_tmp 	   = np.array(y - pred_tmp)
		chi2_tmp   = np.dot(np.dot(r_tmp,np.linalg.inv(N_diag)),r_tmp.T)

		print("\nChi2_tmp is %lf at step %d, chi2 is %lf and lambda is %lf." % (chi2_tmp, i, chi2, lamb))

		if chi2_tmp < chi2:
			print("Good step!!")
			chi2   = chi2_tmp
			r 	   = r_tmp
			pred   = pred_tmp
			params = params_tmp
			Ap, cdtn = Ap_func(pred_func, params)

			print("New Chi2 is %lf and lambda is now %lf." % (chi2, lamb))
			#Arbitrary limit of 0.0001 to how small I allow lambda to get, it just usually seems that 
			#once the algorithm gets to a point where using basic gradient descent seems good and has reduced lambda  
			#past this point, it has to fight back how small it made lambda during its previous successes. 
			#There's probably a way of determining what the best lower limit is based on the magnitude, 
			#variance, and derivatives of the data, but I don't really have the time or knowledge to do that right now.
			if lamb > 0.0001:
				print("Assuming this means we've gotten closer to a solution and decreasing lambda by 10.")
				lamb   = lamb*0.1
			lamb_ctr = 0

			if chi2 - chi2_tmp < chi_tol:
				chi_ctr += 1
			if chi_ctr > 5:
				print("\nNot getting much better anymore, returning parameters values that were settled on.")
				cov_mat    = np.linalg.inv(curv)
				params_err = np.sqrt(np.diag(cov_mat))

				return params, params_err, pred, r, chi2, cov_mat
		elif chi2_tmp >= chi2:
			print("Got chi2 of %lf, benchmark is currently %lf." % (chi2_tmp, chi2))
			print("Assuming that that is because where we are in parameter space is best navigated by just looking at the gradient, augmenting lambda by 10.")
			lamb 	 = lamb*10.0
			lamb_ctr += 1
			chi_ctr  = 0

			if lamb_ctr > 10: #Being generous about the range of values the computer precision can accurately represent
				print("\n\nWell, it seems making lambda bigger isn't helping...")
				print("assuming I increasd lambda because using the curvature wasn't working, not sure what to do, so stopping.")
				cov_mat    = np.linalg.inv(curv)
				params_err = np.sqrt(np.diag(cov_mat))

				return params, params_err, pred, r, chi2


	print("The maximum number of iterations has been attained without exiting.")
	print("This means the solution found was not stable. Either the algorithm has failed, or maybe more steps are required")
	cov_mat    = np.linalg.inv(curv)
	params_err = np.sqrt(np.diag(cov_mat))
	return params, pred, r, chi2




def plot_results(wmap, fit, r, save=False, filename="plot_of_results.png"):

	fig, axes = plt.subplots(2, 1, sharex=True, dpi=255)

	axes[0].set_title("Fit of CMB P_spec Data.")
	axes[0].set_ylabel("Power Spectrum")
	axes[0].errorbar(wmap[:, 0], wmap[:,1], wmap[:,2], fmt=".", label="CMB data", markersize=1, elinewidth=0.5)
	axes[0].plot(wmap[:,0], fit, linestyle="-", label="Fit", markersize=5, linewidth=0.7)
	axes[0].legend()

	axes[1].hlines(0,0.5,1)
	axes[1].scatter(wmap[:,0], r, s=0.5)
	axes[1].set_ylim([min(r)*1.1, max(r)*1.1])
	axes[1].set_xlabel("k")
	axes[1].set_ylabel("Residuals")

	if save:
		plt.savefig(filename)
	plt.show(block=True)

	return




def save_results(fit, params, params_err, ext=""):

	data_filename = "final_fit_" + ext + "data.txt"
	data_file 	  = open(data_filename, "w")
	data_file.write(str(list(fit)))
	data_file.close()

	param_filename = "final_fit_" + ext + "params.txt"
	param_file = open(param_filename, "w")
	param_file.write(str(list(params)))
	param_file.close()

	err_filename   = "final_fit_" + ext + "params_err.txt"
	param_err_file = open(err_filename, "w")
	param_err_file.write(str(list(params_err)))
	param_err_file.close()

	cov_filename = "final_fit_" + ext + "flat_cov_mat.txt"
	cov_file 	 = open(cov_filename, "w")
	cov_file.write(str(list(cov_file.flatten())))
	cov_file.close()

	return




def main():
	pars=np.asarray([65,0.02,0.1,2e-9,0.96,0.05])
	wmap=np.loadtxt('../wmap_tt_spectrum_9yr_v5.txt')

	cdtn_nums = np.zeros(6)
	for scl in np.linspace(200, 260, 7):
		Ap, cdtn = get_Ap(get_spectrum, pars, scl=scl)
		print("Condition number for fraction %d is %lf." % (scl, cdtn))
		cdtn_nums[int(scl/10 - 20)]


	plt.plot(cdtn_nums)
	plt.show()
	plt.clf()


	lamb_init = 0.0001

	### For fit holding tau constant ###################################################################################
	params, params_err, fit, r, chi2, cov_mat = lev_marq(wmap[:,0], [wmap[:,1], wmap[:,2]], pars[:-1], get_spectrum_hold_tau, get_Ap, lamb_init, gauss_N=True)
	print("Results when holding tau at 0.05:")
	print("Final params are: ", params)
	print("Their errors are: ", params_err)
	print("which give a chi2 of: ", chi2)
	print("Plotting fit...")

	#Save data so that I don't have to rerun this every time I want to try a new plot.
	save_results(fit, params, params_err, cov_mat, ext="hold_tau_")
	plot_results(wmap, fit, r, save=True, filename="hold_tau_results.png")
	### Fit holding tau constant (end) #################################################################################


	### For fit with all parameters free ###############################################################################
	print("\nStarting process while including tau.\n")


	params, params_err, fit, r, chi2, cov_mat = lev_marq(wmap[:,0], [wmap[:,1], wmap[:,2]], pars, get_spectrum, get_Ap, lamb_init, gauss_N=True)
	print("Results when fitting all params:")
	print("Final params are: ", params)
	print("Their errors are: ", params_err)
	print("which give a chi2 of: ", chi2)
	print("Plotting fit...")

	save_results(fit, params, params_err, cov_mat, ext="with_tau_")
	plot_results(wmap, fit, r, save=True, filename="with_tau_results.png")
	### Fit with all parameters free (end) #############################################################################



if __name__ == "__main__":
	main()
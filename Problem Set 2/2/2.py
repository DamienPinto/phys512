import numpy as np
import matplotlib.pyplot as plt




def get_cdtn_num(A):
	A 		 = np.matrix(A)
	mtx2 	 = A.transpose()*A
	mtx2_sym = mtx2 + mtx2.transpose()
	es,vecs  = np.linalg.eig(mtx2_sym)
	eabs 	 = np.abs(es)
	cond 	 = eabs.max()/eabs.min()
	return cond




def dec_exp(x, A, B, y0, x0):
	return (A-y0)*np.exp(-(x-x0)/B) + y0




def call_dec_exp(t, params, t0):
	return dec_exp(t, *params, t0)




def dec_exp_ders(x, A, B, y0, x0):
	ders = np.zeros((len(x), 3))

	ders[:, 0] = np.exp(-(x-x0)/B)
	ders[:, 1] = ((x-x0)/B**2)*(A-y0)*np.exp(-(x-x0)/B)
	ders[:, 2] = -np.exp(-(x-x0)/B) + 1.0
	#No x0 because holding constant in this problem.

	return ders




def plot_results(t, y, fit, chi2, r, A, B, y0, x0):
	print("\n")
	print("Chi2 at this point: ", chi2)
	print("Final parameters are:")
	print("\tA: ", A)
	print("\tB: ", B)
	print("\tt0: ", x0)
	print("\ty0: ", y0)
	print("\n")
	print("Plotting final fit...")
	plt.plot(t, y, "k-")
	plt.plot(t, fit, "b--", label="Final Fit")
	plt.xlabel("t")
	plt.ylabel(r"y [W/m^2](?)")
	plt.title("Final Fit Using Newton's Method")
	plt.legend()
	plt.show()
	print("...done.")

	print("Plotting residuals...")
	plt.clf()
	plt.plot(t, r, 'b-')
	plt.xlabel("t")
	plt.ylabel(r"y_{fit} - y")
	plt.title("Residuals of Final Fit Using Newton's Method")
	plt.show()
	print("...done")

	max_err = np.abs(r).max()
	rms 	= np.sqrt(np.mean(r**2))
	print("\n")
	print("The maximum error is of the magnitude: ", max_err)
	print("RMS error is: ", rms)




def main():
	data = np.loadtxt("ps2_data.txt",delimiter=",")
	print(data.shape)
	t_tot = np.array(data[:, 0])
	y_tot = np.array(data[:, 1])


	y0 	   = 1.5
	A 	   = y_tot.max()
	t0 	   = t_tot[y_tot.tolist().index(A)]
	B 	   = 0.1
	params = np.array([A, B, y0])

	#Adjust t and y arrays to only have relevant data.
	t = t_tot[y_tot.tolist().index(A):-400]
	y = y_tot[y_tot.tolist().index(A):-400]

	model = lambda t,params: call_dec_exp(t, params, t0)
	guess = model(t, params)

	plt.plot(t, y, 'k-')
	plt.plot(t, guess, 'b--', label="Guess")
	plt.title("Data vs Guess")
	plt.xlabel("t")
	plt.ylabel(r"Flux (W/m^2)[?]")
	plt.legend()
	plt.show()

	chi2_tmp  = 10000 #Because big.
	tol 	  = 1e-3
	max_iters = 15 #Since in class 5 seemed to do the trick given a reasonable guess.
	up_cntr   = 0

	for i in range(max_iters):
		fit  = model(t, params)
		r 	 = -(fit - y)
		chi2 = np.sum(r**2)
		#Implement dp = ((dA/dm)^2)^-1 * (dA/dm)^T * r to get amount by which to increment 
		#parameters.
		#As in class:
		ders   = dec_exp_ders(t, *params, t0)
		lhs    = np.linalg.inv(np.dot(ders.transpose(),ders))
		rhs    = np.dot(ders.transpose(),r) 
		dp 	   = np.dot(lhs,rhs)
		params = params + dp


		if np.abs(chi2_tmp - chi2) < tol:
			print("Stopping after " + str(i) + " iterations.")
			print("Attained point where progression in parameter-space isn't generating much change in chi2.")
			print("\n")
			print("Condition number of derivatives matrix is: ", str(get_cdtn_num(ders)))
			plot_results(t, y, fit, chi2, r, *params, t0)

			break

		if chi2 > chi2_tmp:
			up_cntr += 1
			if up_cntr > 3:
				print("Three times now Newton's method has made us go in a direction that makes the chi2 worse.")
				print("Assuming something this is the best we're gonna bet despite not respecing the threshold.")
				plot_results(t, y, fit, chi2, r, *params, t0)

				break

		chi2_tmp = chi2


	#Noise of data pre-flare event:
	t_pre 	= y_tot.tolist().index(A)
	std 	= np.std(y_tot[:t_pre])
	print("std: ", std)
	N 	 	= np.identity(131)*std
	curv 	= np.dot(ders.transpose(), ders)
	sig_par = std*np.sqrt(np.diag(np.linalg.inv(curv)))

	print("sig_par: ", sig_par)

	fit_p_err = model(t, params+2*sig_par)
	fit_m_err = model(t, params-2*sig_par)

	plt.plot(t, y, "k-")
	plt.plot(t, fit, "b-")
	plt.plot(t, fit_p_err, "b--")
	plt.plot(t, fit_m_err, "b--")
	plt.show()


if __name__ == "__main__":
	main()
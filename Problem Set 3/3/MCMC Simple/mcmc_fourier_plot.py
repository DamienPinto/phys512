import numpy as np
import matplotlib.pyplot as plt

def get_mcmc_data(filename):
	
	mcmc_file 	 = open(filename, "r")
	mcmc_str  	 = mcmc_file.read()
	mcmc_str  	 = mcmc_str[37:]
	mcmc_str_arr = mcmc_str.split("], [")
	mcmc_arr 	 = np.array([np.fromstring(step.strip("[").strip("]"), sep=", ") for step in mcmc_str_arr])
	print(mcmc_arr.shape)

	return mcmc_arr

def get_fourier_plots(mcmc_data, x):
	title_arr = [r"Hubble Constant $H_0$", r"Physical Baryon Density $\Omega_bh^2$", r"Cold Dark Matter Density $\Omega_ch^2$",
				 "Primordial Amplitude of Fluctuations", "Slope of the Primordial Power Law", 
				 r"Optical Depth $\tau$", r"$\chi^2$"]
	filename_arr = ["H0", "ombh2", "omch2", "As", "ns", "tau", "chi2"]

	for i in range(len(title_arr)):
		title 	 = title_arr[i] + r" Chain"
		filename = filename_arr[i] + "_chain.png"
		fourier_title = title + r" in Fourier Space"
		fourier_filename = "fourier_of_" + filename
		fourier_filename_noDC = "fourier_of_" + filename[:-4] + "_noDC.png"

		plt.clf()
		plt.title(title)
		plt.xlabel("Step Number")
		plt.ylabel("Insert Correct Units Here")
		plt.plot(x, mcmc_data[:, i])
		plt.savefig(filename)
		# plt.show()

		fourier_mcmc = np.abs(np.fft.rfft(mcmc_data[:, i][1:int(len(mcmc_data[:,i])/2)]))

		plt.clf()
		plt.title(fourier_title)
		plt.xlabel(r"$(Number of Steps)^{-1}$")
		plt.ylabel("Insert Correct Units Here")
		plt.loglog(fourier_mcmc)
		plt.savefig(fourier_filename)
		# plt.show()

		plt.clf()
		plt.title(fourier_title)
		plt.xlabel(r"$(Number of Steps)^{-1}$")
		plt.ylabel("Insert Correct Units Here")
		plt.loglog(fourier_mcmc[1:])
		plt.savefig(fourier_filename_noDC)
		# plt.show()

	return




def main():
	mcmc_data = get_mcmc_data("mcmc_chain_simple2.txt")
	x = np.arange(len(mcmc_data))
	get_fourier_plots(mcmc_data, x)




if __name__ == "__main__":
	main()
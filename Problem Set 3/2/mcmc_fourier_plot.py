import numpy as np
import matplotlib.pyplot as plt

def get_mcmc_data(filename):
	
	mcmc_file 	 = open(filename, "r")
	mcmc_str  	 = mcmc_file.read()
	mcmc_str  	 = mcmc_str[37:]
	mcmc_str_arr = mcmc_str.split("], [")
	mcmc_arr 	 = np.array([np.array(step.strip("[").strip("]")) for step in mcmc_str_arr])
	print(mcmc_arr.shape)

	return mcmc_arr
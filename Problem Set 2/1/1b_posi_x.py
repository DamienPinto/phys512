import numpy as np
import matplotlib.pyplot as plt
from lin_mtx_fit_tools import *




def main():
	n = 100000
	x = np.linspace(1e-3, 1e7, n)
	y = np.log2(x)

	x_ch 		  = np.linspace(-1, 1, n)
	x_scaled, exp = np.frexp(x)
	y_scaled 	  = np.log2(x_scaled)
	m_ch, order   = do_lin_mtx_fit(x_ch, (y_scaled*4 -3), get_Cheby_mtx, 1e-6, v=True, cdtn=True)
	y_ch 		  = (m_ch+3)/4 + exp
	y_ch_err 	  = y_ch - y

	y_poly = np.polyval(np.polyfit(x, y, order-1), x)
	y_poly_err = y_poly - y

	plt.plot(x, y, 'k-')
	plt.plot(x, y_ch, 'r--', label="Chebyshev Fit")
	plt.plot(x, y_poly, 'b--', label="np.polyfit")
	plt.title(r"Fits of $log_2(x)$, $x \in [10^{-3}, 10^7]$")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend()
	plt.show()

	#Plot errors:
	plt.plot(x, y_ch_err, 'r--', label="Error on Chebyshev fit.")
	plt.plot(x, y_poly_err, 'b--', label="Error on polyfit.")
	plt.title(r"Errors on $log_2(x)$, $x \in [10^{-3}, 10^7]$, w.r.t. Different Fits.")
	plt.xlabel("x")
	plt.ylabel(r"$y_{fit} - np.log2(x)$")
	plt.legend()
	plt.show()

	#Get max errors:
	print("\n")
	print("Max Error for Chebyshev Fit: " + str(y_ch_err.max()))
	print("Max Error for np.polyfit: " + str(y_poly_err.max()))

	#Get RMS errors:
	rms_ch   = np.sqrt(np.mean(np.array(y_ch_err**2)))
	rms_poly = np.sqrt(np.mean(np.array(y_poly_err)**2))
	print("\n")
	print("RMS Error for Chebyshev Fit: " + str(rms_ch))
	print("RMS Error for np.polyfit: " + str(rms_poly))




if __name__ == "__main__":
	main()
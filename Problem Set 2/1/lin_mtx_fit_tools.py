import numpy as np
import matplotlib.pyplot as plt

'''
Goal: Make set of programs meant to perform a linear fit using matrix methods with various bases (eg. polynomial, 
Legendere, Chebyshev, etc...).

Steps:
	-Get data/data range y.
	-Input y into function that will use it to construct the A matrix using prefered basis.
	-Input A and y into function that applies A in the appropriate way as to apply the best chi^2.
	-Do once with Chebyshev polynomials.
	-Do again with simple polynomials.
	-Compare.
'''




def get_cdtn_num(A):
	mtx2 	 = A.transpose()*A
	mtx2_sym = mtx2 + mtx2.transpose()
	es,vecs  = np.linalg.eig(mtx2_sym)
	eabs 	 = np.abs(es)
	cond 	 = eabs.max()/eabs.min()
	return cond




def get_Cheby_mtx(x, order):
	'''
	Function that takes in data 'x' and a polynomial order 'order' and constructs a 
	(len(x) by 'order') Chebyshev polynomial matrix. 
	'''

	if np.abs(x).max() > 1:
		print("get_Cheby_mtx: The data enterred is outside the valid range for Chebyshev polynomials (-1, 1).")
		quit()
	if order < 1:
		print("get_Cheby_mtx: You entered an order < 1, which means a mistake was made, or that you want just a fit that's a constant.")
		quit()

	mtx 	 = np.zeros((len(x), order+1))
	mtx[:,0] = 1.0
	mtx[:,1] = x

	if order > 1:
		for i in range(1, order):
			mtx[:,i+1] = 2*x*mtx[:,i] - mtx[:,i-1]

	return np.matrix(mtx)




def get_poly_mtx(x, order):
	'''
	Function that takes in data 'x' and a polynomial order 'order' and constructs a 
	(len(x) by 'order') simple polynomial matrix. 
	'''

	if order < 1:
		print("get_poly_mtx: You entered an order < 1, which means a mistake was made, or that you want just a fit that's a constant.")
		quit()
	mtx 	 = np.zeros((len(x), order+1))
	mtx[:,0] = 1.0

	for i in range(1, order+1):
		mtx[:,i] = mtx[:,i-1]*x

	return np.matrix(mtx)




def lin_mtx_minim(A, d):
	'''
	Function that takes as input *numpy matrices* A and d where d is the data to be fit and transformation matrix A
	constructed using a certain basis (polynomial, Legendre, Chebyshev, etc...).
	Casts them to numpy matrices.
	Implements m = (A^T * A)^-1 * (A^T * x) <- assumes noise matrix N == I.
	'''

	#As done in class:
	lhs  = np.linalg.inv((A.transpose()*A))
	rhs  = A.transpose()*(d.transpose())
	m    = lhs*rhs
	pred = A[:,:-1]*m[:-1]

	return np.array(pred)[:,0]




def do_lin_mtx_fit(x, y, poly_fctn, tol, v=False, cdtn=False):
	
	#Putting 10 bc/ in slide 11, lect. 4 from class condition number goes from bad to ridiculous around there.
	for i in range(1, 10):
		A 	  	= poly_fctn(x,i)
		y_mdl 	= lin_mtx_minim(A, np.matrix(y))
		max_err = np.abs(y - y_mdl).max()

		print("v: ",v)
		if v:
			print("At order " + str(i) + " , max error is: " + str(max_err) + ".")

		if cdtn:
			#Get condition number of matrix, as shown in class:
			cond = get_cdtn_num(A)
			print("Condition number of matrix used for linear fit is: " + str(cond))

		#If the largest error is smaller than the tolerance, return the fit data and report the order.
		if max_err < tol:
			return y_mdl, i

	print("Didn't meet error threshold of " + str(tol) + " but passed a 10th order fit, so might want to revise method used...")
	return y_mdl, i




def main():
	
	#Get log2 data from x in (0.5, 1] to model Chebyshev with.
	x = np.linspace(0.5, 1, 100)
	y = np.log2(x) #What log2(x) should actually look like per numpy 

	#Get Chebyshev fit for tolerance of 1e-6:
	#Learned that the Chebyshev method requires x from -1 to 1 for the creation of the 
	#polynomial matrix during a very long and painful process...
	x2 = np.linspace(-1, 1, 100) 
	y_ch, order = do_lin_mtx_fit(x2, y, get_Cheby_mtx, 1e-6, v=True, cdtn=True)
	y_ch_err = y_ch - y


	#Get polynomial modelling for same tolerance:
	y_poly = np.polyval(np.polyfit(x, y, order-1), x)
	y_poly_err = y_poly - y

	#Plot errors:
	plt.plot(x, y_ch_err, 'r.', label="Error on Chebyshev fit.")
	plt.plot(x, y_poly_err, 'b--', label="Error on polyfit.")
	plt.title("Errors on log2(x), x in (0.5, 1], w.r.t. Different Fits.")
	plt.xlabel("x")
	plt.ylabel("y_fit - np.log2(x)")
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
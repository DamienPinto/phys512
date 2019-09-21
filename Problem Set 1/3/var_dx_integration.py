import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

DEPTH = 0


def sin(x):
	return np.sin(2*x)




def gaussian(x):
	return  np.exp(-0.5*(x-2)**2/1**2)/(1*np.sqrt(2*np.pi))




def integral_poly_o2(y, dx):
	return (y[0] + 4*y[1] + y[2])/3*dx




def integrate_me(fctn, x_i, x_f, tol, y = np.array([])):
	if x_i==x_f:
		print("Start and end-points of integration are same. Returning zeros.") #Assuming you didn't mean this and quiting.
		return 0.0, 0.0, 0.0

	x  	   = np.linspace(x_i, x_f, 5)
	dx 	   = np.diff(x)[0]
	n_eval = 0 #Tracker of number of evaluations made by this 
	slctn  = np.array([0, 2, 4])

	n_eval_class = 0 #Number of evaluations that would have been made using the function written in class.

	if type(y) != type(np.ndarray([])):
		y = np.array(y)
		
	#If the length of the input array isn't exactly 3, assume they didn't structure it with the exact usage of this
	#function in mind and evaluate the 5 points.
	if len(y) != 3:
		y = fctn(x)
		n_eval += len(y)
	else:
		y = np.insert(y, 1, fctn(x[1]))
		y = np.insert(y, 3, fctn(x[3]))
		n_eval += 2
	n_eval_class += 5

	f1  = integral_poly_o2(y[slctn], 2*dx)
	f2  = integral_poly_o2(y[0:3], dx) + integral_poly_o2(y[2:], dx)
	err = np.abs(f2-f1)
	# print("err: ", err)
	# print("tol: ", tol)
	# print("err < tol: ", err<tol)
	# print("num_calls: ", num_calls)

	if err < tol:
		num_calls = 0
		return (16*f2 - f1)/15, err, n_eval, n_eval_class, num_calls
	else:
		yr = y[2:]
		yl = y[0:3]

		num_calls = 2
		area_r, err_r, n_eval_r, n_eval_class_r, num_calls_r = integrate_me(fctn, x[2], x[4], tol/2.0, yr)
		area_l, err_l, n_eval_l, n_eval_class_l, num_calls_l = integrate_me(fctn, x[0], x[2], tol/2.0, yl)

		n_eval += n_eval_r + n_eval_l
		err 	= err_r + err_l
		area 	= area_r + area_l

		n_eval_class += n_eval_class_r + n_eval_class_l
		num_calls 	 += num_calls_l+num_calls_r

		return area, err, n_eval, n_eval_class, num_calls




def main():
	#														  fctn 		x_i x_f  tol
	area, err, n_eval, n_eval_class, num_calls = integrate_me(gaussian, -3,  7, 1e-10, y = np.array([]))
	scipy_val = integrate.quad(gaussian, -3, 7)
	print("Area calculated by my function: %r +/- %r" % (area, err))
	print("Scipy calculated and area: %r +/- %r" % (scipy_val[0], scipy_val[1]))
	print("My function evaluated f %d times." % n_eval)
	print("The function written in class would have evaluated f %d times." % n_eval_class)
	print("My function called itself (including the initial call that we technically made, not it) %d times." % (num_calls+1))


	# x = np.linspace(-3, 7, 1000)
	# y = gaussian(x)

	# plt.plot(x, y)
	# plt.show()




if __name__ == "__main__":
	main()
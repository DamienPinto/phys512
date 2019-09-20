import sys
import numpy as np
import matplotlib.pyplot as plt




def get_params(filename):

	print("Getting parameters. Opening file...")
	
	file 	 = open(filename)
	file_str = file.read()
	file.close()

	'''
	Params presumed to be 
	pow_i pow_f np
	a dtype
	
	Where the pows are the powers of 10 at which to start and end checking dx = 10^pow
	np is the number of steps with which to span pow_i -> pow_f
	a is the constant multiplying x in the argument of the exponent.
	dtype is the datatype to use when storing/manipulating data.
	'''
	params 	  = [ _.split(" ") for _ in file_str.split("\n")[:-1]]
	print("Params:", params)
	params[0] = [float(_) for _ in params[0]]
	params[1] = [float(params[1][0]), np.dtype(params[1][1])]

	print("Parameters obtained.")

	return params




def get_2side_der(y, dx, n):

	y_roll    = np.roll(y, -int(2*n))
	#Doing (y_roll - y)/(2*n*dx) creates the derivative.
	#For example (y_roll[0] - y[0])/(2*n*dx) = (f(x_2n) - f(x_0))/(2n*dx) = f'(x_n)
	y_der_num = (y_roll - y)/(2*n*dx)
	#Cut off the last 2*n because they consist of nonsensical stuff like f(x_(N-2n))-f(x_0) and that
	y_der_num = y_der_num[:int(-2*n)]

	return y_der_num




def get_composite_2side_der(y, dx, dtype):
	
	y_2side_der_1dx = get_2side_der(y, dx, np.float32(1).astype(dtype))
	y_2side_der_2dx = get_2side_der(y, dx, np.float32(2).astype(dtype))

	#The array of derivatives made using points 1 dx away from x will have two more points than the array of derivatives 
	#made using points 2*dx away from x so need to trim those off
	y_2side_der_1dx = y_2side_der_1dx[1:-1]

	y_composite_der = (4*y_2side_der_1dx - y_2side_der_2dx)/3

	return	y_composite_der




def main():
	#Open parameter file and get params
	argv = sys.argv
	[[p_i, p_f, n_p], [a, dtype]] = get_params(argv[1])
	a = np.float64(a).astype(dtype)

	#Get all the p with which we want to try dx = 10^p
	p_range = np.linspace(p_i, p_f, int(n_p))
	print("p_range: ", p_range)

	err_list 	 = []

	for i in range(len(p_range)):
		dx = np.float64(10**p_range[i]).astype(dtype)

		#Generate x and y for true function
		x = np.array([1-2*dx, 1-dx, 1.0, 1+dx, 1+2*dx])
		y = np.exp(a*x, dtype=dtype)

		#generate true f'
		y_der_t = a*y
		#(I guess this will only deal with f(x) = c*exp(ax) for now since that's the relevant example and you kind of 
		#need to know the true derivative of the function you're examining beforehand)

		#Compute numerical composite derivative of f for the current dx
		y_composite_der = get_composite_2side_der(y, dx, dtype)
		# print("Size of y_composite_der: ", len(y_composite_der))

		#Compute mean error for this dx
		err = np.abs(y_der_t[2] - y_composite_der)
		err_list.append(err)
		print("Obtained error for " + str(i+1) + " p out of " + str(len(p_range)))


	plt.plot(p_range, err_list)#, p_range, n_xstep_list)
	plt.show()




if __name__ == "__main__":
	main()
PS1_1.pdf:
	Analytic derivation of delta and estimation of optimal values for delta.

check_composite_der_perf.py:

	Call: python3 check_composite_der_perf.py params.txt


	Description of function:
	This program will access params.txt and extract parameters for pow_i, pow_f, np, a, and dtype.
	It will use pow_i, pow_f, and np to generate a range of exponential powers of 10 to cycle through for dx.
	Ex: If pow_i = -7, pow_f = 1 and np = 100, then dx will start at 10^-7, then 10^-6.9292..., 10^-6.8585... until, in
	100 iteration, it has reached dx = 10^0.

	At every iteration it creates the x array where x = [1-2*dx, 1-dx, 1.0, 1+dx, 1+2*dx], then generates
	y = np.exp(a*x) -> this is the "a" retrieved from params. It then creates the "true" derivative y_der_t = a*y, and
	the composite we derived using get_composite_2side_der(). Given our number of points, only the derivative at 
	x = 1.0 will be computed. The absolute difference between this and y_der_t[2] = y'(x=1.0) is then computed and
	added to a list.

	Once this deifference for every dx = 10^p is computed and stored in a list, this list is then plotted.

	
	Output:
	A matplotlib plot of the difference between the computed derivative and the true derivative of y = exp(a*x)
	at x = 1.0.

	float32_exp(x)_-7to0_zoomed.png: plot for p in [7, 0], y = exp(x) and the datatype used during the computation being
	of 32-bit precision. A significant zoom has been applied to show the minimum at p ~ -1.5, dx ~ 0.03, which is very 
	comparable to the analytically predicted value.

	float64_exp(x)_-7to0_zoomed.png: Shows the same thing as float32_exp(x)_-7to0_zoomed.png but with 64-bit precision
	data-storage.

	Similarly float32_exp(0.01x)_-7to0_zoomed.png, float64_exp(0.01x)_-7to0_zoomed.png show minimums at dx ~ 10^0.5 
	and dx ~ 10^-1.25 respectively, whic line up approximately with the analytic predictions as well.


params.txt:
	Parameter file that check_composite_der_perf.py calls on.

	Format:
	pow_i pow_f np
	a dtype

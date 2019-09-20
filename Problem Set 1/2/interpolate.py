import numpy as np
import matplotlib.pyplot as plt




def check_mono(arr):
	'''
	Function that returns a number indicating whether a series of elements in an array are monotonic increasing,
	monotonic decreasing, constant, or not any of those.
	'''
	if np.all(np.diff(arr)) >= 0:
		if not (np.all(np.diff(arr)) > 0):
			print("check_mono: Checked array and it is monotonic increasing but at least two adjacent elements have the same value.")
		return 1
	elif np.all(np.diff(arr)) <= 0:
		if not (np.all(np.diff(arr)) < 0):
			print("check_mono: Checked array and it is monotonic decreasing but at least two adjacent elements have the same value.")
		return 2
	elif np.all(np.diff(arr)) == 0:
		print("check_mono: Checked array is constant.")
		return 3
	else:
		return 0




def get_adj_indicies(arr, val):
	'''
	Function that returns the two indicies of the elements in arr who's values "sandwich" the val quantity.  
	'''

	if np.amin(arr[1:]) > val or np.amax(arr[:-1]) <= val:
		print("get_adj_indicies: Value given is not contained within the valid range for interpolation in the array provided. Quitting because the whole point of this question needs that to be true to continue.")
		quit()
	elif val in arr:
		print("get_adj_indicies: Value given is already present in provided array, returning indicies adjacent to that element.")
		idx = np.where(arr==val)[0][0]
		return [idx-1, idx+1]
	else:
		indicies = np.abs(arr-val).argsort()[:2]
		return indicies




def create_interp_table(x, y, M, order):
	print("Creating interpolation table...")
	if M%2 == 0 and M > 0:
		n_b4 = M/2
		n_af = M/2
	elif M > 2 and M%1 == 0:
		n_b4 = (M-1)/2
		n_af = (M+1)/2
	else:
		print("create_interp_table: So far you've told me to interpolate of 0, 1, or a negative amount of points. Probably want to reassess that.")
		quit()

	func_list = []

	for i in range(1, len(x)):
		p_fit = np.polyfit(x[int(i-n_b4):int(i+n_af)], y[int(i-n_b4):int(i+n_af)], order)
		func_list.append(p_fit)
	print("...done.")
	return func_list




def get_lakeshore_data():
	print("Obtaining lakeshore data...")
	file = open("lakeshore.txt", "r")
	file_str = file.read()
	file.close()
	file_str_array = file_str.split("\n")

	T 	 	   = []
	V 	 	   = []
	dV_dT_site = []

	for row in file_str_array[:-1]:
		#print("Row: ", row)
		row = row.replace("\t\t", "\t").split("\t")
		#print("Row2: ", row[2])
		T.append(float(row[0]))
		V.append(float(row[1]))
		dV_dT_site.append(float(row[2]))

	V = np.array(V)
	T = np.array(T)
	dV_dT_site = np.array(dV_dT_site)
	dV_me = (np.roll(V, -1) - V)[:-1]
	dT_me = (np.roll(T, -1) - T)[:-1]
	#Made my own derivative because it seemed like the one provided by the site wasn't always correct.
	dV_dT_me = (dV_me*1000)/dT_me 
	print("...done.")
	return [T, V, dV_dT_site, dV_dT_me]




def so_3xWatchya_VanT(trgt_V):
	[T, V, dV_dT_site, dV_dT_me] = get_lakeshore_data()
	mono = check_mono(V)
	if mono == 0:
		print("Independant data is not monotone, which is required for cubic interpolation. Please sort it and come back.")
		quit()

	adj_idcs = np.sort(get_adj_indicies(V, trgt_V))
	strt_idx = adj_idcs[0]-1
	end_idx  = adj_idcs[1]+1

	relevant_V   = V[strt_idx:end_idx]
	relevant_T   = T[strt_idx:end_idx]
	interp_table = create_interp_table(relevant_V, relevant_T, 3, 3)

	low_interp  = np.polyval(interp_table[0], trgt_V)
	high_interp = np.polyval(interp_table[1], trgt_V)
	val 		= (low_interp+high_interp)/2
	err 		= np.abs(low_interp - high_interp)

	print("\nInterpolated T value for V = %f V is: T = (%f +/- %f)K" % (trgt_V, val, err))

	return val, err





def main():
	[T, V, dV_dT_site, dV_dT_me] = get_lakeshore_data()
	mono = check_mono(V)
	if mono == 0:
		print("Independant data is not monotone, which is required for cubic interpolation. Please sort it and come back.")
		quit()
	interp_table = create_interp_table(V, T, 3, 3)#Hardcoded Interpolation with 3 points and order 3 polynomials.

	#Also harcoded to check 10 times as many Ts as provided in the valid range
	more_V = np.linspace(V[1], V[-2]-1e-5, 10*len(V[1:-1]))
	more_T = np.zeros(len(more_V))
	err_T  = np.zeros(len(more_V))
	for i in range(len(more_V)):
		tbl_idx   = get_adj_indicies(V, more_V[i])[0]
		poly 	  = interp_table[tbl_idx]
		val 	  = np.polyval(poly, more_V[i])
		more_T[i] = val

		#Calculating the error by checking what the adjacent interpolations would get and avergaing if possible.
		err = 0
		err_cnt = 1
		if i != 0:
			poly_m1	  = interp_table[tbl_idx-1]
			val_m1 	  = np.polyval(poly_m1, more_V[i])
			err 	  = np.abs(val - val_m1)
			err_cnt  += 1
		if i < len(interp_table)-1:
			poly_p1	  = interp_table[tbl_idx+1]
			val_p1 	  = np.polyval(poly_p1, more_V[i])
			err 	  = (err+np.abs(val - val_p1))/err_cnt
		err_T[i] = err


	plt.plot(V, T, "*")
	plt.plot(more_V, more_T, color="#3F7F4C")
	plt.fill_between(more_V, more_T-err_T, more_T+err_T,
    alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99',
    linewidth=0)

	plt.show()





if __name__ == "__main__":
	main()
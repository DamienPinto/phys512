interpolate.py:
	
	so_3xWatchya_VanT(trgt_V): Function that takes as input a voltage and, if it's within the valid interpolation range
							   of the lakeshore data, then it will return the interpolated T value.

							   It makes two interpolations of third order polynomial based on three points and returns
							   the average. It gives a rough estimate of the error based on the difference in the values
							   returned by both those interpolations. The rationality there is that both interpolations
							   are supposed to be over intervals that include the desired V and T(V), so the difference
							   between the two should be related to the stability/confidence of the interpolation.


	call ing interpolate.py itself: Will plot T vs V with an interpolated line and error for the whole range of data for
									which interpolation is possible. (The error is there, you just have to zoom until
									you can see a shaded area) To do this, the program...

	Will access the lakeshore.txt data, check that its V data is monotonic, use np.polyfit to get the coefficients of all the unique 3rd order polynomials that can be done from 3 consecutive points, store each set of those coefficients in a table.

	It then generates a linspace for a new set of T that will be 10x larger than the initial T, goes through each point, determines the best interpolated polynomial coefficients to use/retrive from the table and uses polyval to generate a V values for said point.

	The error is estimated by the average of the differences between our point evaluated with its relevant polynomial and evaluated using both polnomials adjacent to that one in the table.

	It then plots this using matplotlib. The error is very small but can be seen if one zooms in en
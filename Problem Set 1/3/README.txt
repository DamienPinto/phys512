integrate.py:

	integrate_me():Function that follows the algorithm that we outlined in class but it passes the reusable evaluations
				 of the function it's integrating "down the rabbit hole" such that it does not have to start from scratch at each recursive step. 

				 Default running it from the console as "python3 integrate.py" will make it evaluate a normalized 
				 gaussian curve from -3 < x < 7 with an error tolerance of 1e-10, and then compare with the result
				 scipy.integrate.quad would obtain for the same integral.

				 If you would like to change this, go to line 86 (Hopefully I marked the relevant parameters clearly
				 enough, if not... sorry :S). 

				 If N is the total number of function-to-be-integrated calls made during the whole integration process and n is the number of times the integration function was called (either by itself or the initial one by us), and both are starting from a blank slate (as in no already-computed f(x) values), then the
				 function in class would obey: N = 5*n and mine would obey N = 2*n + 3.
				 This is because my function passes along 3 of the still-relevant evaluations of f(x) it has every time
				 it must go down to a deeper recursion level (5*n -> 3*n), however, when starting from scratch, it must
				 still evaluate f(x) 5 times (+3). 
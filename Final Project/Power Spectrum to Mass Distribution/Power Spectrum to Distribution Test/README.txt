This is a directory where I test my functions that are made to get the power spectra of distributions (p_spec_test.py/make_power_spectrum),
and also produce distributions from a power spectrum (p_spec_test.py/make_fourier_space).


To run code: python3 p_spec_test.py
NOTE: Will take a little while, approx. 5mins.


The first part of the code produced a 3D distribution of size 100x100x100 that represents a sin(kr) wave,
where k = pi and r is the distance of a point from the ceneter of the space.
A slice of it can be seen in init_test_dist.png,
and a slice of its Fourier transformation can be seen in ft_init_test_dist.png.
We can see that, although we wanted to produce a "perfect" sin wave, the ring in fourier space a t k = pi is
messy and there is activity in close-by modes due to the our data not being continuous.

Then, the power spectrum of said distribution is taken and plotted and can be seen in p_spec_init_test_dist.png.

This power spectrum is then input into make_fourier_space to generate a Fourier space that would *approximately* respect said power spectrum.
This is only approximate because the power spectrum, when taken, elliminates locality of information, meaning that there is a sort of degeneracy:
many different distributions can produce the same power spectrum. Given this, a (pseudo)randomized process is used to generate a distribution that,
if it were infinitely large, would produce the same power spectrum, but due to the finite-ness of our, well... everything? (but I guess specifically
computation methods) we can only ensure a distribution that will produce a power spectrum with similar features. Which is usually ok because
general deatures is usua;;y what we're interested in.

A slice of the produced Fourier space can be seen in ft_new_dist.png. As we can see, the most activity can be seen in a ring of radius pi,
which makess sense given our initial distribution. We can also see however that the boundary of the ring in a lot less definite
than in the fourier transformation of our initial distribution. This is due to the afformentionned necessary randomness.
If the program performed correctly, then the average of the power in the inner rings should average out, and the net power in the
ring at k = pi should be relatively high.

And we can see in p_spec_new_dist.png that that is the case, and that the power spectrum of the new distribution indeed resembles that of the initial distribution.

If we compare the two distributions at face value (init_test_dist.png and new_dist.png), they look fairly different and like they would not have similar statistical features,
but the size of the dominant/most frequent structures in bost distributions are of similar magnitude, it's just that the sin wave is much more ordered.
The placement of adjacent phases is much more ordered, and this is what is lost when taking the complex norm during the construction of the power spectrum.


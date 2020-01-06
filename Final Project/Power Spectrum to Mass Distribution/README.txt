code for this number: python3 nbody.txt

NOTE: You should read through the "README.txt" in the "Power Spectrum to Distribution Test" directory
      before going through this as it outlines the method of distribution generation used here.
      
NOTE: Code will initialize space and universe class but exit right before actually starting the simulation.
      This is to a)Save time. and b)Ensure that my initializations weren't being altered before I checked them.
      
NOTE: Once I initialize a density distribution, I then decompose it into a random amount of masses that would
      generate said distribution. I haven't found a quick way to do this, so it takes a little while...
      Due to this, I reduced the size of the space being simulated to a 16x16x16 box. This means the initialization
      takes about 5mins instead of... years. (Which honeslty might have been the case with the 100x100x100 space)
      
NOTE: Most of the pictures/plots I have for the 16x16x16 example I also have for the 100x100x100 case.
      These can be found in the "100x100x100 Attempt" directory. The only ones missing are the ones that start
      with "uni_..." as, even running overnight, I was only able to deal with less than 20000 of the 1000000 points of
      the density distribution.


So first, I write a function for a power spectrum that comprises of a decaying exponential centered at 0
and a Gaussian curve centered at a certain k-value. The plot of tis can be seen in "initialized_dist_p_spec.png"
(and the same example for the larger distribution can be seen in the "100x100x100 Attempt" directory)

I then generate a distribution using that power spectrum using the same method outline in the
"Power Spectrum to Distribution Test" directory. A slice of this distribution can be seen in
"init_rho_slice.png", and it's power spectrum, taken as verification, can be seen in
"init_dist_p_spec.png". A smoothed version of this power spectrum can be seen in
"init_dist_p_spec_smthd.png".

Given that the value of every cell in the density distribution indicates the *total* mass at that point
and not the *number* of massesa at that point, we must decompose these values into multiple masses.
Since this can be done in an infinite number of ways, a random process is used that generates a number
of masses for each point. The average masse of these masses is arbitrarily chosen, but in such
a way as to ensure that there are not something like 100000 masses per point.
An efficient way to do this was not found and so this is achieved using three "for" loops.
This takes a sizable amount of time and so the size of the distribution was reduxed to 16x16x16

Once the masses and their positions are determined, they are submitted into the initialization of the universe class.
Print statements verify that the first mass in both the distributions *before* the universe class, and *inside* the
universe class, are the same, indicating that some order has been mainained.
Despite this, once the universe class is initialized, if we check its density distribution vs that of the
initialized/constructed density distribution, we see that they are not the same. ("uni_rho_check.png")
Further more, taking the power spectrum of the density distribution inside the universe class reveals one
that is flat, and so that corresponds to a Gaussian noise distribution.
("uni_rho_p_spec.png" and "uni_rho_p_spec_smthd.png") This rules out a simple transpose of coordinates
or something during transcirption. Something much more disruptive occurs somewhere after initialization of
the desired density distribution and the initialization of the universe class, but that ws not found here.

All in all, I have shown a way of constructing a density distribution that corresponds to a desired power spectrum,
and *theoretically* this could be used in my nbody simulation, I have just failed somewhere in the bookkeeping
of said distribution.

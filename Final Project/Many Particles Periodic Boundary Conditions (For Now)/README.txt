Comments on simulation: 

The simulations seems to be behaving as far as I can see. The particles coalesce into smaller structures close to where they were initialized before all converging towards the center of gravity. The problem that can be seen after is that a large number get "flung" off, which I would attribute to me not setting the "fuzzy-ness" of the particles (the profimity after which the force between two particles is set to a constant) high enough. A larger fuzz_dist value when initializing the space would have lead to smaller forces being generated and thus less of and aftershock.


Comments on energy:

The energy scales here are clearly off. I thought I was computing the energy in terms of pixel/index values so I divided by the spatial length, but it turns out I had already incorporated that into my computation of the potential kernel by including a dx term. I removed that term for subsequent simulations but this one took a day to run so I'm afraid I won't have much timje to run it again. The energy plot contains 1000 steps and only 260 are included here because of github's 25Mb data cap on uploads. THe behaviour after the 260 snapshots included here is fairly the same as in snapshots 245-260 though. 

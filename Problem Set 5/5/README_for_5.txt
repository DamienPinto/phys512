So I tried implementing the temperature by adding jut 1.0 to every pixel along the first face in the x-direction. I do this at every iteration of the conjugate gradient method conj_grad (or at least believe I do), by passing the argument "side" through and doing "results = results + side". This can be seen, I believe, on line 110, or really the first line of the for loop in conj_grad().

ALthough I seem to have adequately created a cube, it does not seem like I was successful in making the temperature change in any way, which is surprising since I am adding 1.0, which is at the upper range of what the simulation can usually produce, i.e. this +1.0 along a whoole face should quickly dominate the value range.

I tried plotting the line from the heated end of the box to the non-heated one in "line_T.png", however it does not look totally correct. I would that guess I have something wrong in how I initialize my box?

Some contants you have to set at the beginning of this problem is the thickness of the box and, if you wanted to be realistic about it, something resembling a heat capacity. Maybe as a multiplier that modulates any change the amount by which the value at a "box" pixel changes from one iteration to another.

I might have done this wrong, it's super late and I eally need sleep... :P

Thank you for the semester marking our assignments, helping us with tuorials, and being all-round pleasant people :)

Happy holidays and best of luck in your own projects!

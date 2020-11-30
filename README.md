# V-to-mu_Demo
 
This small java swing app demonstrates the core principle of Sparsey's algorithm for approximately preserving the similarity of inputs to the similarity of their codes.  In this case, the codes are modular sparse distributed codes (MSDCs), which are sparse binary codes, and the code similarity metric is just intersection size, since all codes are constrained to have the same weight (i.e., the same number of 1's).  The MSDC code is as follows.  The coding field (CF) consists of Q WTA modules, each with K binary units, and a code is just a choice one winner in each of the Q modules.  Thus all codes are of weight, Q.

The core principle is extremely simple.  All you have to do is add noise proportional to an input's novelty into the process of choosing its code.  

Because of the use of MSDCs, an input's novelty, or to be precise, its inverse, which I call "familiarity" and denote, G, can be computed extremely quickly, in fact, with constant time complexity.  G is simply the average of the max V values in the Q CMs. Since the architecture is fixed for the life of the system, the number of steps needed to compute G remains constant as additional inputs are stored.  

A unit's V value is just a normalized version of its input summation.  Again, since the architecture is fixed, the number of steps needed to compute a unit's V value remains constant for the life of the system, as does the number of steps needed to compute the V values of all Q x K units comprising the CF.

Computing the amount (power) of the noise to be added to the process of choosing winners also takes a constant number of steps, and in fact can easily be pre-computed and stored as a table.  Actually adding a noise sample to each Q x K units' V values also has constant time complexity.

The final selection of the winner in a module is done by transforming the V values of the units into prob (rho) values that reflect the noise and making a random draw from the rho distribution.  This also takes afixed number of steps (proportional to the log of the number of units, K, in a module). 

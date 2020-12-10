# V-to-mu_Demo
 
How to build and run this java app.

1. Clone project to a local repo.
2. The cloned project folder (by default named, V-to-mu_Demo) will have a subfolder, nbproject.  The presence of that subfolder will allow you to open
   the project as a project in Netbeans (probably analogous situation for other IDEs). 
3. Start Netbeans, and click on button to open a project.
4. Navigate to folder V-to-mu_Demo in the "Open Project" dialog and open the project.
5. If you have the latest Netbeans, or more specifically, if your JDK is Java 11 or higher, you should be able to compile and run the app.
6. If not, rt click on the V-to-mu_Demo project (in Netbeans Project window) and select "Properties", which opens the Properties dialog
7. At bottom of the dialog, click on the "Src/Binaries" pulldown button and select the highest java src you have avaailable.  It should run if for at least
   Java 8 and higher.
   
In netbeans, when you build, you might see that there are problems with the build, even though it runs.  Specifcally, it might be that there are two libraries specified, which you can see if you open the Properties dialog and click on "Libraries".  You should be able to just remove the two libs, "AbsoluteLayout.jar" and "wwing-layout-1.0.4.jar".  I thought these libs were necessary to be in the distribution, but apparently not.  I'll get rid of this comment here in the Readme when I resolve this either way.  But for now, just removing the two libs should work for you.

When you run the app, the window opens.  You see four main panels. Many of the controls or labels have tooltips which explain their operation.  You can also click on the "Instructions" toolbar button and a window with instructions opens.  At the most general level, the way you use the app is as follows.

1. When the app opens there will be a default number, Q, of WTA competitive modules (CMs) in the Mac panel (lower right).
   And there will be a default number, K, of units per CM.  Current defaults are Q=8 and K=4. The lower left panel shows 
   a larger view of the first (leftmost) CM in the Mac panel.  It's shown by itself really just to show what's going on 
   in a single CM more clearly, especially if a lot of units have been added.
   
2. You can use the K spinner to adjust the number of units per CM  Also, note the V values of the units are initially
   all equal, but will generally be quickly overidden once you begin playing with the controls.
   
3. Now you can click on the "Generate New Sample" button. Each click generates a new pattern of V inputs in all Q CMs.  
   In each CM, there will be one randomly chosen unit to have a V value equal to the current setting of of the "Max V" 
   slider (which is tied to the "Global Familarity" slider. And, in each CM, the other K-1 units will receive a random V 
   value chosen from the range determined by the current settings of the Min and Max Crosstalk sliders. 
   
4. Each time you hit the "Generate New Sample" button, you will get a new draw determined as described in the previous bullet.
   You will also see the Expected and Actual Accuracy values updated.  The idea here is that since we assign a unit
   in each CM with the max V, that set of Q max V cells is to be interpreted as the code of the most similar stored item 
   in the Mac. We don't actually maintain an explicit set of stored items.  Rather, the crosstalk simulates the effects of
   stored inputs. Nevertheless, given this interpretation, we can consider the max V cell in each CM to be the correct winner, 
   i.e., the cell that should ultimately be chosen winner in each CM.  However, the whole point of Sparsey's learning algorithm 
   is that we don't simply pick the max V cell in each CM.  Rather we transform the V values to a probability distribution 
   (within each CM) and choose the winner from the distribution. The transform, V-to-mu (which after normalizing the mu values, 
   is really a V-to-rho transform), depends on the global familiarity, G. 
   
5. When G is near 1 (which must mean there is at least one unit in each CM with a V near 1, hence the tying of the two 
   sliders), that means the input is highly familiar and therefore that we should want the max V cell to win in all 
   (or at least, most) of the CMs, which would correspond to activating the code of the familiar (i.e., previously
   experienced) input.  You can see how playing with the various sliders controlling the transform affects the expected 
   accuracy, i.e., the expected fraction of CMs in which the max V cell wins. On the other hand, when G is near 0 [which 
   means all units (in each CM) have near-zero V values], that indicates that the input is highly unfamiliar, in which case, 
   we should want to assign a highly unique code to the input.  Thus, low G causes the V-to-mu transform to flatten, i.e., 
   causing the V values to be compressed toward the same low value, thus yielding a near uniform rho distribution in each CM, 
   which in turn, leads to the minimum (chance-level) expected interseciton of the chosen code to any previously stored codes.

This small java swing app demonstrates the core principle of Sparsey's algorithm for approximately preserving the similarity 
of inputs to the similarity of their codes.  In this case, the codes are modular sparse distributed codes (MSDCs), which are 
sparse binary codes, and the code similarity metric is just intersection size, since all codes are constrained to have the same 
weight (i.e., the same number of 1's).  The MSDC code is as follows.  The coding field (CF) consists of Q WTA modules, each with 
K binary units, and a code is just a choice one winner in each of the Q modules.  Thus all codes are of weight, Q.

The core principle is extremely simple.  All you have to do is add noise proportional to an input's novelty into the process 
of choosing its code.  

Because of the use of MSDCs, an input's novelty, or to be precise, its inverse, which I call "familiarity" and denote, G, can be 
computed extremely quickly, in fact, with constant time complexity.  G is simply the average of the max V values in the Q CMs. 
Since the architecture is fixed for the life of the system, the number of steps needed to compute G remains constant as additional 
inputs are stored.  

A unit's V value is just a normalized version of its input summation.  Again, since the architecture is fixed, the number of steps 
needed to compute a unit's V value remains constant for the life of the system, as does the number of steps needed to compute 
the V values of all Q x K units comprising the CF.

Computing the amount (power) of the noise to be added to the process of choosing winners also takes a constant number of steps, 
and in fact can easily be pre-computed and stored as a table.  Actually adding a noise sample to each Q x K units' V values 
also has constant time complexity.

The final selection of the winner in a module is done by transforming the V values of the units into prob (rho) values that reflect 
the noise and making a random draw from the rho distribution.  This also takes afixed number of steps (proportional to the log of the 
number of units, K, in a module). 

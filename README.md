# MissiePlots

This repository contains a varity of codes to reproduce plots I have made. Please reference this repo if you use it in your work.  

Tutorials and example plots for each of the plotting functions can be found in the tutorials folder and documentation can be found in the docs folder.  

If you have seen a nice plot in my work (papers, posters, talks etc) and the code for it isn't here, let me know (sophia.vaughan@physics.ox.ac.uk) and I will see if I can add it.  

## Installing

This code is presently not avalible on pip so you will have to clone the repository and add it to the system path to use it like an installed package. To do so, first, in the command line, do the following:  

```
cd <the folder which you are going to add to the path>
git clone git@github.com:SophiaVaughan/MissiePlots.git
```

This clones the repository into the current folder. You should now see a folder called MissiePlots within that dierctory. Then at the begining of the python code in which you want to use MissiePlots:  

```
import sys
sys.path.insert(0, "/path/to/the/folder/MissiePlots_is_in")
```

It then should be possible to import the code as normal:  

```
from MissiePlots.<a module> import <some function>
```

Note that this will then allow you to import any other code in the directory you added so if you have other git repositories in there it will be possible to use those to. It is also possible to permanently add the directory to the system path but this is left as an exercise for the reader (google it).  

## Dependecies

While the code will likely work for other versions of these, it has not been tested on them.  

python 3.8.10  
numpy 1.17.4  
matplotlib 3.1.2  

## FAQ

*Who is Missie?*  
Look in the docs folder.  

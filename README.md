windprofile_characterization: GLWind Wind Profile Characterization
====================================

Author: Elliott Walker (walker.elliott.j@gmail.com)

Extension of the old [glwind_codebase](https://github.com/windysensors/glwind_codebase/), focusing on the [wind profile analysis/characterization project](https://engineering.csuohio.edu/glwind_reu/wind-profile-characterization-based-surface-terrain-and-atmospheric-thermal-stability) for the [GLWind program](https://engineering.csuohio.edu/glwind_reu/glwind_reu). Currently being worked on at Texas Tech University as a continuation of work from Summer 2024.

Required Python packages
-----------------------
numpy
pandas
matplotlib
seaborn
tqdm
argparse
multiprocessing
scipy
windrose

Why is this a mess?
-----------------------
Up until now I've just been throwing stuff together to see what works and get an idea of what kind of things I want this to be capable of. I now have a decent idea, and so I'll be combining everything in a hopefully easier, more managable, and *more useful* way.

Structuring
-----------------------
Code up until now is in `src`. Most of the stuff in `src/old` isn't really useful anymore. I'll note that the `newsonic.py` is not currently functional. Right now, `combine.py` puts the KCC data together, `reduce.py` does basic QC and then computes useful values, and `plots.py` (as well as `roses.py` for some wind rose stuff) does a strange mix of analysis and plotting. `sonic.py` is being updated into `newsonic.py` to do an okay job handling analysis of ultrasonic data, but this is even more of a WIP. `helper_functions.py` has a variety of common statistical, scientific, or purely convenience functions used by any and all of the others.

I'm now putting things together in `new`. Ideally this will replace `src` at some point. Within it (structuring and naming subject to much change) is: `new/lib` containing files with common functions - this is basically `helper_functions.py` split up into smaller chunks; `prepare.py` to do lots of what `reduce.py` did; `kcc.py` as an example of how everything else will be configured and called from a single file which must on its own do the standardization done by `combine.py`; `analyze.py` with analysis functionality; `plotting.py` with plotting functionality.

Right now the code is set up to assume a structure of the parent directory containing  `windprofile_characterization`. Data is assumed to be in a folder called `data` on the same level, and outputs from running `src` are put into one called `outputs` also on that same level. For my convenience as I fix things up and slowly consolidate everything into the nice neat new package, outputs from `new` are put into a folder `results` at that level rather than mushed together into `outputs`.

If for some unlikely reason you are using this while I'm working on it, feel free to contact me at the above email with questions.

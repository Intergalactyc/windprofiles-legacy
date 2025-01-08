WindProfiles: GLWind Wind Profile Characterization
====================================

Author: Elliott Walker (walker.elliott.j@gmail.com)

Extension of the old [glwind_codebase](https://github.com/windysensors/glwind_codebase/), focusing on the [wind profile analysis/characterization project](https://engineering.csuohio.edu/glwind_reu/wind-profile-characterization-based-surface-terrain-and-atmospheric-thermal-stability) for the [GLWind program](https://engineering.csuohio.edu/glwind_reu/glwind_reu). Currently being worked on at Texas Tech University as a continuation of work from Summer 2024.

Required Python packages
-----------------------
numpy
pandas
matplotlib
tqdm
scipy
windrose

Structuring
-----------------------
Code up until now is in `src_old/`. Most of the stuff in `src_old/old` isn't really useful anymore. I'll note that the `newsonic.py` is not currently functional. Right now, `combine.py` puts the KCC data together, `reduce.py` does basic QC and then computes useful values, and `plots.py` (as well as `roses.py` for some wind rose stuff) does a strange mix of analysis and plotting. `sonic.py` is being updated into `newsonic.py` to do an okay job handling analysis of ultrasonic data, but this is even more of a WIP. `helper_functions.py` has a variety of common statistical, scientific, or purely convenience functions used by any and all of the others.

I'm now putting things together in the `windprofiles` package. Ideally this will replace `src` at some point. Within it are the main modules, as well as a subpackage `lib/` containing files with common functions - this is basically the old `helper_functions.py` split up into smaller chunks.

Analysis using the `windprofiles` package is in the `analysis` directory: `kcc.py` handles the analysis of the Cedar Rapids KCC met tower data.

If for some unlikely reason you are using this while I'm working on it, feel free to contact me at the above email with questions.

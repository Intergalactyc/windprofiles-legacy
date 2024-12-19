windprofile_characterization: GLWind Wind Profile Characterization
====================================

Author: Elliott Walker (walker.elliott.j@gmail.com)

Extension of [glwind_codebase](https://github.com/windysensors/glwind_codebase/) which will only focus on the [wind profile analysis/characterization project](https://engineering.csuohio.edu/glwind_reu/wind-profile-characterization-based-surface-terrain-and-atmospheric-thermal-stability) for the [GLWind program](https://engineering.csuohio.edu/glwind_reu/glwind_reu).

This will be where I put my work for Fall 2024, especially that towards improving the poster for the November 2024 APS DFD meeting in SLC.

This is kind of a mess at the moment, I need to clean stuff up.

Required Python packages
-----------------------
numpy
pandas
matplotlib
seaborn
tqdm
argparse
multiprocessing

Usage
------
Run the files from the `src` directory.

The "slow data" analysis should be performed first. Run `python combine.py` to generate first-pass cleaned and formatted slow data (`combined.csv`). Then run `python reduce.py` to do the actual analysis on the result, providing output `ten_minutes_labeled.csv`. Once this has been done, plots may be generated and sonic analysis may be done. (Output files will be within the `outputs` directory)

Plotting is done using the `plots.py` and `profiles.py` scripts; in the `main` entry points change what functions are called to decide what to plot.

Sonic analysis can be done without the slow data (data matching must be disabled!) but the full analysis requires both the `combined.csv` and `ten_minutes_labeled.csv` results from the slow analysis. Multiprocessing is supported: pass argument `-n <number of processors>` to determine how many CPUs are used. Example usage:
``` 
python sonic.py -c -n 8 --data="../../data/KCC_FluxData_106m_SAMPLE/" --target="../outputs/sonic_sample/" --match="../outputs/slow/ten_minutes_labeled.csv" --slow="../outputs/slow/combined.csv"
```

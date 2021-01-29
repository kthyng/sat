# sat

This software reads in satellite data that is gathered at Dr. Chuanmin Hu's website: https://optics.marine.usf.edu/. 

## Installation

This assumes you are using Anaconda to access Python, and you have already [installed Python 3 via Anaconda](https://www.anaconda.com/products/individual).

1. Clone repository: `git clone git@github.com:kthyng/sat.git`
1. Move into new repo directory: `cd sat`
1. Create environment with necessary packages: `conda create --name SAT python=3.8 --file requirements.txt`

## Run script


## Examples

To run examples, also install `jupyterlab` with 
> conda install jupyterlab

The basic run of the script is
> run plot_sat [year] [var] [area] [figarea]

where each [] should be filled in with the value and  the year is given as an integer, the field to plot `var` is given as a string, the name of the area the satellite data is from on the Hu website `area` is given as a string, and the name of the figure area to plot `figarea` is given as a string. A basic example is

> run plot_sat 2016 "rgb" "galv" "galv_plume"

Options are as follows:
* `var`: Variables that are available at https://optics.marine.usf.edu/ that this software has previously been setup to read in. "sst" (sea surface temp) or "oci" (chlorophyll-a with good correction algorithm) or "ci" (chlorophyll-a with no sun glint) or "rgb" (color) or "CHL" (chlorophyll-a)'
* `area`: Names of available geographic areas at https://optics.marine.usf.edu/, for where to extract data. "gcoos" (full Gulf of Mexico) or "wgom" (western Gulf of Mexico) or "galv"
* `figarea`: The area of the resulting replotted figure. Currently available options are "wgom" (western Gulf of Mexico) or "txla" (TXLA domain) or "galv_plume" or "galv_bay" or "TX".

Information can also be found in the terminal window with 
> python plot_sat.py --help

or in the ipython window with
> run plot_sat --help

1. SHOW DIFFERENT INPUT OPTIONS




TODO:
- have a mode to show a single plot
- show the different figareas in a notebook
- DEMO DIFFERENT INput options
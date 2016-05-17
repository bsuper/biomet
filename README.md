# Applying machine learning techniques to CO2 and CH4 fluxes from Fluxnet towers using CIMIS, MODIS, LANDSAT, and other external data

### Files

`utils.py` - Utility functions for Pandas and for processing Fluxnet, CIMIS, MODIS, Landsat, and other external data sources. These files are given by Housen Chu for this project. The data are located in the `data/` folder.

`exp.py` - Contains methods for featurizing data and for merging various data sources together for specific experiments.

`regression.py` - Contains methods for pre-processing featurized data for ML techniques, methods to perform cross-validation with the available models, methods to train and predict, and methods to visualize results. Available models: Random Forests, Gradient Boosted Trees, SVM, and Neural Network.

`plot_corr.py` - Methods for plotting the correlation matrix from the data

`data/` - folder containing the data files mentioned above

`input/` - [DEPRECATED] folder containing data files used for earlier runs

`notebooks/` - Contains iPython notebooks showing results from runs
* `notebooks/correlation_plots.ipynb` - Shows correlation heatmap
* `notebooks/cv_{target}.ipynb` - [DEPRECATED] Cross validation scores from early runs on target
* `notebooks/cv_{source tower}-{target tower}_{target}.ipynb` - Cross-site cross-validation using source tower data to predict target tower data
* `data.ipynb` - An overview of the data files
* `plot_interpolation.ipynb` - Shows the interpolation used to transform 8/16/32-day data into daily data
* `predict-{source years}_{target years}-{target}.ipynb` - Use 1 year of data to predict other years
* `{target tower .e.g wp}_{target}_{8_day (250m MODIS) | landsat | none (500m MODIS)}_{cv | predict}.ipynb` - Main notebooks showing cross validation and prediction results with nice plots using the available models.

### Running the code locally

Once you've cloned the folder, `cd` into the folder:

1. `git submodule init & git submodule update` Downloads the data for this project
* `virtualenv venv` Creates a virtual environment called venv. Put briefly, a virtual environment allows you to contain all the dependencies needed for a project within one environment.
- `source venv/bin/activate` Enter the virtual environment
3. Install tensorflow [[see this link]](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#pip-installation)
4. `pip install -r requirements.txt` Install all the necessary dependencies in the virtual environment`
5. `jupyter notebook` Start the iPython notebook
6. Browse the files!

Perform `git pull origin master & git submodule update` after the above steps routinely to update your code/data to the most recent version.

### Running the code in a VM

If you have Windows or you don't want to setup your computer for local run, I've setup a VirtualBox image for you guys to use. It's quite large (3.4 GB), but campus wifi should make it take download a lot faster!

[Download VM](https://gitlab.com/bsuper/biomet-vm-2/raw/master/biomet-ipython.ova)

Instructions

1. [Download Virtualbox](https://www.virtualbox.org/wiki/Downloads)
2. Open Virtualbox. `File > Import Appliance` and select the downloaded .ova file.
3. `cd biomet`
4. `git pull origin master` # fetch latest version
5. `git submodule update` # update data folder
6. `jupyter notebook` # open ipython notebooks

user: biomet

pass: biomet123!

### Known issues
* Some of the code reference the `input/` folder, which is deprecated. To make those files work, try changing the reference to `input/` to `data/`. If that doesn't work, then, unfortunately, the file is quite old and probably would require rewriting to work. Sorry!

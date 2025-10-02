# Code Repository for Master Thesis
## Hard- and Software developments towards simultaneous 1H/129Xe imaging on a benchtop MRI system

The code created within the framework of this master’s thesis can be classified into two categories. With the first category being python based simulations and data evaluation functions, collected in a module rf_functions.py for RF (radio frequency) specific functions and in a module for image evaluation image_functions.py. The second category contains MATLAB based scripts, which rely on Pure Devices' openMATLAB toolbox to control the MRI system. The MATLAB scripts can only be used with the openMATLAB toolbox and serve the purpose to calibrate the custom double resonant probe and perform basic Spin Echo imaging sequences. 

To run the python code it is recommended to clone the entire repository and follow the package versions in requirements.txt for the python installation. The repository contains additional Jupyter Notebook (.ipynb) files and example measurement data that illustrate the application of the main functions for data evaluation and RF simulations.

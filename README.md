# Ising Model Simulation

This project simulates the 2D Ising Model using Glauber and Kawasaki dynamics.

## File Structure
* `raman_visual.py`: Contains the logic for collecting and modifying Raman data.
* `plot_raman.py`: Script to plot the heat map.
* `test.py`: Test script.

## To Plot (`plot_raman.py`)
Visualise the Ising model using Glauber of Kawasaki dynamics.

**Arguments:**
* `-p`, `--path`: Path to folder which contains the raman data.
* `-x`, `--x`: Number of grid points in x.
* `-y`, `--y`: Number of grid points in y.
* `-m`, `--mode`: Choose viewing mode: 'sliding_spectra' or 'whole_spectra'.

**Example Command:**
python3 plot_raman.py -p ~/Code/Data_SH/FullCavity_20x20_2umsteps -x 20 -y 20 -m sliding_spectra
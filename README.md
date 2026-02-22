# Ising Model Simulation

This project simulates the 2D Ising Model using Glauber and Kawasaki dynamics.

## File Structure
* `raman_visual.py`: Contains the logic for collecting and modifying Raman data.
* `test.py`: Test script.

## To Plot (`plot_raman.py`)
Visualise the Ising model using Glauber of Kawasaki dynamics.

**Arguments:**
* `-p`, `--path`: Path to folder which contains the raman data.
* `-x`, `--x`: Number of grid points in x.
* `-y`, `--y`: Number of grid points in y.

**Example Command:**
python plot_raman.py ~/Code/Data_SH/FullCavity_20x20_2umsteps 20 20 --pipeline 2 --shift-range 50
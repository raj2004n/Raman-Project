# Raman Project

In progress..

## File Structure
```text
## Project Structure

```text
.
├── notebooks/             # Experimental scripts and analysis sandboxes
│   ├── cluster.py
│   ├── unmix.py
│   └── raman_helper.py    # Utility functions for Raman spectroscopy
├── outputs/               # Generated figures and model results
│   ├── mineral_map.png
│   └── training_results.png
├── scripts/               # Top-level executable scripts
│   └── plot_raman.py
├── src/                   # Core source code
│   ├── analysis/          # Endmember estimation and signal processing
│   ├── cnn/               # Neural network architecture, training, and evaluation
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── data/              # Data loading, grid processing, and I/O
│   └── visualisation/     # Plotting themes and spatial mapping tools
│       ├── theme.py
│       └── view_predict.py
├── README.md
└── requirements.txt       # Project dependencies
```
## To Plot (`plot_raman.py`)
...

**Arguments:**
* `-p`, `--path`: Path to folder which contains the raman data.
* `-x`, `--x`: Number of grid points in x.
* `-y`, `--y`: Number of grid points in y.
...

**Example Command:**
python3 plot_raman.py ~/Code/Data_SH/FullCavity_20x20_2umsteps 20 20 --pipeline 2 --shift-range 50
import numpy as np
from raman_helper import *
import matplotlib.pyplot as plt
from pathlib import Path 
import numpy as np
import pandas as pd
import ramanspy as rp
"""
spectral_data = np.random.rand(50, 50, 1500)
spectral_axis = np.linspace(100, 3600, 1500)
raman_image = rp.SpectralImage(spectral_data, spectral_axis)
ax = rp.plot.image(raman_image.band(1500))
rp.plot.show()
"""

"""

Okay so their plot outputs a single slice of some chosen band

Plan potentially is to use their raman image object for data preperatation, and then use that to plot the area under curve
I mean I just want some way to plot the entire spectra, which i can do, and then add a interactive bar to choose band (which it would load)
This approach avoids integrating, also if i can use their object was data prep, and then get the data back out of the container
that would be great

Alternatively, can store all the plots of from bands, and then collect those plots and try show that with interactibility

Don't have to necessarily use their plots, I can use their baseline correction stuff,
and then plot that data myself

What if i print out the raman image object?

What information does the raman spectra object hold?


Okay, the data can at least be viewed using raman_image.__dict__

Data can be accessed using raman_image.spectral_axis for eg

This means that i can use their program to perfrom denoising and such, 
and then use my own plotting code to plot

I guess simplest approach is loop throuhg each file
for that file create a spectra object
- denoise
get the denoised intensities
plot it myself

lets try:
"""
path = Path("~/Code/Data_SH/FullCavity_20x20_2umsteps").expanduser() # store as Path object for easier manipulation
files = list(path.glob('*.txt')) # extract .txt files and store as list

if not files:
    print("No .txt files found in that directory.")
else:
    raman_shifts = pd.read_csv(
        files[0],
        sep='\t',
        names=['raman_shift'],
        header=None,
        usecols=[0]
        )['raman_shift'].tolist()

    file = files[0] # only take the first file, as we would handle in a loop

    intensity_arr = pd.read_csv(
            file, 
            sep='\t', 
            names=['intensity'], 
            header=None,
            usecols=[1],
            )['intensity'].tolist()

    

    # make into rp spectra object
    raman_spectra = rp.Spectrum(intensity_arr, raman_shifts)

    # denoise 
    savgol = rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3)
    gaussian = rp.preprocessing.denoise.Gaussian()

    # baseline correction methods
    asla = rp.preprocessing.baseline.IARPLS()
    iasls = rp.preprocessing.baseline.IASLS()
    airpls = rp.preprocessing.baseline.AIRPLS()
    drpls = rp.preprocessing.baseline.DRPLS()
    iarpls = rp.preprocessing.baseline.IARPLS()
    aspls = rp.preprocessing.baseline.ASPLS()

    baseline_corrections = [asla, iasls, airpls, drpls, iarpls, aspls]

    for baseline_correction in baseline_corrections:
        spectra = baseline_correction.apply(raman_spectra)
        rp.plot.spectra(spectra, title=f"{baseline_correction}")
        rp.plot.show()
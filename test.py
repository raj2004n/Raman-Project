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
path = Path("FullCavity_20x20_2umsteps") # store as Path object for easier manipulation
files = list(path.glob('*.txt')) # extract .txt files and store as list

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

#plt.plot(raman_shifts, intensity_arr)
#plt.show()
# make into rp spectra object
raman_spectra = rp.Spectrum(intensity_arr, raman_shifts)
# plot
rp.plot.spectra(raman_spectra)
rp.plot.show()
# denoise
# yo i have to choose this? ok then
savgol = rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3)
# apply denoise
raman_spectra = savgol.apply(raman_spectra)
rp.plot.spectra(raman_spectra)
rp.plot.show()
# baseline correction
baseline_corrector = rp.preprocessing.baseline.IARPLS()
# apply baseline correction
raman_spectra = baseline_corrector.apply(raman_spectra)
rp.plot.spectra(raman_spectra)
rp.plot.show()
# normalisation
vector_normaliser = rp.preprocessing.normalise.Vector()
# apply normalisation
raman_spectra = vector_normaliser.apply(raman_spectra)
rp.plot.spectra(raman_spectra)
rp.plot.show()


# extract data
intensity_arr_new = raman_spectra.spectral_data
raman_shifts_new = raman_spectra.spectral_axis # same as before
plt.plot(raman_shifts, intensity_arr_new)
plt.show()

plt.plot(raman_shifts_new, intensity_arr_new)
plt.show()
# compare original to new

"""

ram = Raman_Data("FullCavity_20x20_2umsteps", 20, 20)
integrals = ram.get_integrals()

fig, ax = plt.subplots(figsize=(8,8))
map = ax.imshow(integrals)
fig.colorbar(map, ax=ax)   
"""
#plt.show()
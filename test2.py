import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from statannotations.Annotator import Annotator
from ai_denoise import *
import ramanspy as rp
from pathlib import Path 
from raman_helper import *

SEED = 19

matplotlib.rc_file_defaults()
plt.rc('font', size=16)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

METRICS = ['MSE', 'SAD', 'SID']
#colors = list(plt.cm.get_cmap()(np.linspace(0, 1, 4)))

def nn_preprocesing(spectral_data, wavenumber_axis):
    flat_spectral_data = spectral_data.reshape(-1, spectral_data.shape[-1])

    output = net(torch.Tensor(flat_spectral_data).unsqueeze(1)).cpu().detach().numpy()
    output = np.squeeze(output)

    output = output.reshape(spectral_data.shape)

    return output, wavenumber_axis

nn_denoiser = rp.preprocessing.PreprocessingStep(nn_preprocesing)

def get_results(spectrum_to_denoise, target, denoiser):

    # Normalise input and output to 0-1
    spectrum_to_denoise = minmax.apply(spectrum_to_denoise)
    target = minmax.apply(target)

    output = denoiser.apply(spectrum_to_denoise)

    metrics_result = {metric: getattr(rp.metrics, metric)(output, target) for metric in METRICS}

    return output, metrics_result

# Load pretrained model
net = ResUNet(3, False).float()
net.load_state_dict(torch.load(r"ResUNet.pt", map_location=torch.device('cpu')))

baseliners = {
    'SG (2, 5)': rp.preprocessing.denoise.SavGol(window_length=5, polyorder=2),
    'SG (2, 7)': rp.preprocessing.denoise.SavGol(window_length=7, polyorder=2),
    'SG (2, 9)': rp.preprocessing.denoise.SavGol(window_length=9, polyorder=2),
    'SG (3, 5)': rp.preprocessing.denoise.SavGol(window_length=5, polyorder=3),
    'SG (3, 7)': rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
    'SG (3, 9)': rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
}

minmax = rp.preprocessing.normalise.MinMax()

# e.g. use

def add_normal_noise(spectrum, std=0.15):
    spectrum = rp.preprocessing.normalise.MinMax().apply(spectrum)

    # add noise
    noise = np.random.normal(0, std, len(spectrum.spectral_data))
    noisy_spectrum = rp.Spectrum(spectrum.spectral_data + noise, spectrum.spectral_axis)

    return noisy_spectrum

path = Path("~/Code/Data_SH/FullCavity_20x20_2umsteps").expanduser()

raman_visual = Raman_Data(path, 20, 20)
raman_slice = raman_visual.get_slice()
raman_slice.flat

selected_index = np.random.randint(0, raman_slice.shape[0])
selected_target = raman_slice[selected_index]
selected_input = add_normal_noise(selected_target)

#print(selected_index)
#print(selected_target.spectral_data.shape)
#print(selected_input.spectral_data.shape)

"""
I suspect that the neural networks requests a certain dimension, due to what it was trained on.

If that is the case then i will have to look into training my own model

That's fine. In that case I can spend the day finishing the other part of the project.
And the reason i was getting weird slices was because the denoise still left bumps, I think.

Try to crop the data so that it is 500 long
"""
spectral_data = selected_input.spectral_data
flat_spectral_data = spectral_data.reshape(-1, spectral_data.shape[-1]) # this makes the data (1, 1340), but the code expects (500, 500)
# so it changes the last things shape into the last of the spectral data, but i thought that this has shape of 1
print(spectral_data.shape)
print(flat_spectral_data.shape)
#nn_results = get_results(selected_input, selected_target, nn_denoiser)[0]
"""

np.random.seed(SEED)

selected_index = np.random.randint(0, raman_slice.shape[0])
selected_target = raman_slice[selected_index]
selected_input = add_normal_noise(selected_target)

nn_results = get_results(selected_input, selected_target, nn_denoiser)[0]
baseline_results = get_results(selected_input, selected_target, baseliners['SG (3, 9)'])[0]

results = minmax.apply([selected_input, baseline_results, selected_target, nn_results])
labels = ['Input (data with noise)', 'Savitzky-Golay (3, 9)', 'Target (authentic data)', 'Neural network']

plt.figure(figsize=(10, 4), tight_layout=True)
ax = rp.plot.spectra(results, plot_type='single', ylabel='Normalised intensity', title='Transfer dataset')
ax.legend(labels)

plt.show()
"""
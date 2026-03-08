import os
import numpy as np
import ramanspy as rp
from collections import defaultdict
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder

def get_data():
    path = os.path.expanduser('~/Code/Data_SH/unrated_unoriented')
    # returns spectral container, contextual data
    return rp.datasets.rruff(path, download=False)

def standardise_data(spectra_list, target_length, x_min=100, x_max=1800):
    standardised_data = []
    # new wavenumber axis the intensity will be interpolated to
    new_x = np.linspace(x_min, x_max, target_length)
    # get all standardised data
    for spectrum in spectra_list:
        x = spectrum.spectral_axis
        y = spectrum.spectral_data
        # interpolator
        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)
        # new interpolated intensities
        new_y = f(new_x)
        # clip the new_y to avoid intensity being below 0
        new_y = np.clip(new_y, 0, None)
        # normalise
        new_y /= (new_y.max() + 1e-8) # plus small value to avoid div by 0 error
        # append new y as a column as required by CNN model
        standardised_data.append(new_y.reshape(-1, 1))
    # returns shape (target_length, 1)
    return np.array(standardised_data)

def augment_shift(spectrum):
    intensity = spectrum[:, 0].copy() # again, the shape is (target_length, 1)

    # generate some random shift integer val between -5 and 5
    shift = np.random.randint(-5, 6)
    # shifts = 0 do nothing
    while shift == 0:
        shift = np.random.randint(-5, 6)

    # use numpy roll to shift for its quickness
    intensity = np.roll(intensity, shift)
    # np roll wraps, replace values to 0 that were wrapped around
    if shift > 0:
        intensity[:shift] = 0
    else:
        intensity[shift:] = 0

    # reshape to fit CNN model's req
    spectrum = intensity.reshape(-1, 1)
    return spectrum

def augment_noise(spectrum):
    intensity = spectrum[:, 0].copy()
    noise = np.random.normal(0, 0.05 * np.abs(intensity))
    # clip to stay bounded between 0 and inf
    intensity = np.clip(intensity + noise, 0, None)
    # normalise
    intensity /= (intensity.max() + 1e-8)
    # reshape to fit CNN model's req
    spectrum = intensity.reshape(-1, 1)
    return spectrum

def augment_linear_combinations(spectra_for_class, n_combinations):
    augmented = []
    n = len(spectra_for_class)
    
    if n < 2:
        return augmented # can't combine single spectrum
    
    for _ in range(n_combinations):
        # random coeff that sum up to one
        coeffs = np.random.dirichlet(np.ones(n))

        combined_intensity = sum(c * s[:, 0] for c, s in zip(coeffs, spectra_for_class))
        combined_intensity /= (combined_intensity.max() + 1e-8)
        augmented.append(combined_intensity.reshape(-1, 1))
    
    return augmented

def build_augmented_dataset(x_train, y_train_labels, n_shift, n_noise, n_combinations):
    augmented_x = []
    augmented_y = []

    # collect the spectras grouped by classes
    # running hash map appraoch
    class_spectra = {}
    for spectrum, label in zip(x_train, y_train_labels):
        if label not in class_spectra:
            class_spectra[label] = []
        class_spectra[label].append(spectrum)
    #TODO: admit mistake that augmentation was not fully correct, try to augment with n_s, n_n = 1, 1
    for label, spectra_list in class_spectra.items():
        for spectrum in spectra_list:
            # add orginal
            augmented_x.append(spectrum)
            augmented_y.append(label)

            # augment by shift
            for _ in range(n_shift):
                augmented_x.append(augment_shift(spectrum))
                augmented_y.append(label)

            # augment by noise
            for _ in range(n_noise):
                augmented_x.append(augment_noise(spectrum))
                augmented_y.append(label)
            
        # augment by linear combinations
        combos = augment_linear_combinations(spectra_list, n_combinations)
        for combo in combos:
            augmented_x.append(combo)
            augmented_y.append(label)
    
    return np.array(augmented_x), np.array(augmented_y)

def leave_one_out_split(y_data):
    # since dictionary can not hold duplicate keys, use that to hold index
    class_indices = defaultdict(list)
    for i, label in enumerate(y_data):
        class_indices[label].append(i)
    
    train_index = []
    test_index = []

    for label, indices in class_indices.items():
        if len(indices) == 1:
            train_index.extend(indices)
        else:
            test_pick = np.random.choice(indices)
            test_index.append(test_pick)
            train_index.extend([i for i in indices if i != test_pick])
    
    return np.array(train_index), np.array(test_index)
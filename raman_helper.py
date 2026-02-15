from pathlib import Path 
from scipy.integrate import simpson
from ai_denoise import *
import re
import numpy as np
import pandas as pd
import ramanspy as rp
import matplotlib
import matplotlib.pyplot as plt

class Raman_Data:
    def __init__(self, path, x, y):
        self.path = path
        self.x = x
        self.y = y

    def get_file_number(self, path):
        """
        Returns the number of the corresponding file.

        eg. ".../FullGrid[1](-18000, -320, -14675)" -> 1

        If no name found, then returns None to signal 
        """
        match = re.search(r'\[(\d+)\]', path.name)
        return int(match.group(1)) if match else None

    def get_files(self):
        path = Path(self.path) # store as Path object for easier manipulation
        files = list(path.glob('*.txt')) # extract .txt files and store as list

        # if no files found
        if not files:
            print("No .txt files found in that directory.")
            return 
        
        # sort the files in the order of their file number
        files.sort(key=self.get_file_number)

        return files
    
    def apply_preprocessing(self, spectra):

        preprocessing_pipeline = rp.preprocessing.Pipeline([
            #TODO: the cropper may want to take the region i am considering
            #TODO cropping does not work because the range of the raman shifts for when initialisign spectral image need to be the same, and my spectra range is different
            #rp.preprocessing.misc.Cropper(region=(500, 1800)),
            rp.preprocessing.despike.WhitakerHayes(),
            rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
            rp.preprocessing.baseline.ASLS(),
            rp.preprocessing.normalise.MinMax(pixelwise=False),
        ])

        return preprocessing_pipeline.apply(spectra)
   
    def get_integrals(self):
        
        # get files
        files = self.get_files()

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()
        
        # list to hold area under curve
        integrals = np.zeros(shape=(self.x, self.y))

        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)

        for file in files:
            # read the intensity column and store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None, usecols=[1],)['intensity'].tolist()
            
            # make raman spectra object and preprocess spectra
            raman_spectra = rp.Spectrum(intensity_arr, raman_shifts)
            raman_spectra = self.apply_preprocessing(raman_spectra)

            # get area under whole curve
            integral = simpson(intensity_arr, raman_shifts)

            # assign integral to its grid position
            integrals[cur_x, cur_y] = integral

            # grid assigning logic
            if cur_y == self.y - 1 and step != -1: # on right boundary
                cur_x -= 1 # step up
                step *= -1 # flip step direction
            elif cur_y == 0 and step != 1: # on left boundary
                cur_x -= 1 # step up
                step *= -1 # flip step direction
            else: # not on boundary
                cur_y += step # step 

        return integrals, raman_shifts
    
    def get_integral_slices(self, slices):
        # get files
        files = self.get_files()

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()

        # intensities, raman shifts (might not need this, depends on future applications)
        #spectral_data = np.zeros(shape=(self.x, self.y, len(raman_shifts))) # matrix of x, y, each element holding a spectra
        #spectral_axis = np.array(raman_shifts)

        # list to hold area under curve
        intensity_slice = np.zeros(shape=(self.x, self.y, len(raman_shifts))) 
        integral_slice = np.zeros(shape=(self.x, self.y, len(raman_shifts) // slices)) 

        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)

        # define methods
        baseline_corrector = rp.preprocessing.baseline.IARPLS()
        savgol = rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3)
        vector_normaliser = rp.preprocessing.normalise.Vector()
  
        for file in files:
            # read the intensity column, store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None,usecols=[1],)['intensity'].tolist()
            
            # make raman spectra object to use rp methods
            raman_spectra = rp.Spectrum(intensity_arr, raman_shifts)
            
            # apply
            raman_spectra = baseline_corrector.apply(raman_spectra)
            raman_spectra = savgol.apply(raman_spectra)
            #raman_spectra = vector_normaliser.apply(raman_spectra)

            # extract the modifed data (for that pixel)
            intensity_arr = raman_spectra.spectral_data
            
            #TODO: Use their cropper
            i = 0
            while i < len(raman_shifts) // slices:
                # eg. integral from index 0 -> 9
                integral = simpson(intensity_arr[i:i+slices], raman_shifts[i:i+slices])
                integral_slice[cur_x, cur_y, i % slices] = integral
                i += 10

            # get area under whole curve
            integral = simpson(intensity_arr, raman_shifts)

            intensity_slice[cur_x, cur_y] = intensity_arr # assign slice

            # grid assigning logic
            if cur_y == self.y - 1 and step != -1: # on right boundary
                cur_x -= 1 # step up
                step *= -1 # flip step direction
            elif cur_y == 0 and step != 1: # on left boundary
                cur_x -= 1 # step up
                step *= -1 # flip step direction
            else: # not on boundary
                cur_y += step # step 

        return integral_slice, intensity_slice, raman_shifts
    
    def get_slice(self):
        # get files
        files = self.get_files()

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()

        spectral_data = np.zeros(shape=(self.x, self.y, len(raman_shifts))) 
        spectral_axis = np.array(raman_shifts)
        
        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)
  
        for file in files:
            # read the intensity column, store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None,usecols=[1],)['intensity'].tolist()
            
            # make raman spectra object to use rp methods
            raman_spectra = rp.Spectrum(intensity_arr, raman_shifts)
            raman_spectra = self.apply_preprocessing(raman_spectra)
            
            # extract the modifed data (for that pixel)
            intensity_arr = raman_spectra.spectral_data
            
            # assign integral to its grid position
            spectral_data[cur_x, cur_y] = intensity_arr

            # grid assigning logic
            if cur_y == self.y - 1 and step != -1: # on right boundary
                cur_x -= 1 # step up
                step *= -1 # flip step direction
            elif cur_y == 0 and step != 1: # on left boundary
                cur_x -= 1 # step up
                step *= -1 # flip step direction
            else: # not on boundary
                cur_y += step # step 
        # return slice
        return rp.SpectralContainer(spectral_data, spectral_axis)
    
    def get_slice500(self):
        # get files
        files = self.get_files()

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()

        spectral_data = np.zeros(shape=(self.x, self.y, 500)) 
        spectral_axis =  np.array(raman_shifts[0:500])
        
        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)
  
        for file in files:
            # read the intensity column, store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None,usecols=[1],)['intensity'].tolist()
            intensity_arr = intensity_arr[0:500]
            # make raman spectra object to use rp methods
            raman_spectra = rp.Spectrum(intensity_arr, spectral_axis)
            raman_spectra = self.apply_preprocessing(raman_spectra)
            
            # extract the modifed data (for that pixel)
            intensity_arr = raman_spectra.spectral_data
            
            # assign integral to its grid position
            spectral_data[cur_x, cur_y] = intensity_arr

            # grid assigning logic
            if cur_y == self.y - 1 and step != -1: # on right boundary
                cur_x -= 1 # step up
                step *= -1 # flip step direction
            elif cur_y == 0 and step != 1: # on left boundary
                cur_x -= 1 # step up
                step *= -1 # flip step direction
            else: # not on boundary
                cur_y += step # step 
        # return slice
        return rp.SpectralContainer(spectral_data, spectral_axis)
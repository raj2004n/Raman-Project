from pathlib import Path 
from scipy.integrate import simpson
import re
import numpy as np
import pandas as pd
import ramanspy as rp

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
    
    def get_raman_image_OLD(self):
        path = Path(self.path) # store as Path object for easier manipulation
        files = list(path.glob('*.txt')) # extract .txt files and store as list

        #TODO an actual way of stopping the program here
        # if no files found
        if not files:
            print("No .txt files found in that directory.")
            return 
        
        # sort the files in the order of their file number
        files.sort(key=self.get_file_number)

        # read one file, only the raman_shift column, choose that raman_shift column and convert tolist
        #read from only one file (since raman_shift shared across all files)
        raman_shifts = pd.read_csv(
            files[0],
            sep='\t',
            names=['raman_shift'],
            header=None,
            usecols=[0]
            )['raman_shift'].tolist()
        
        # list to hold intensities
        intensities = []

        # intensities, raman shifts
        spectral_data = np.zeros(shape=(self.x, self.y, len(raman_shifts))) # matrix of x, y, each element holding a spectra
        spectral_axis = np.array(raman_shifts)
        
        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        shift = 1 # intially steps forward (right)

        for file in files:
        # read the intensity column (1)
        #TODO must be a better way to read this in
        #TODO merge this with loop below
            intensity_arr = pd.read_csv(
                file, 
                sep='\t', 
                names=['intensity'], 
                header=None,
                usecols=[1],
                )['intensity'].tolist()
            # add df to list of df_list
            intensities.append(intensity_arr)

        for intensity_arr in intensities:
            # assign intensity to grid spot
            spectral_data[cur_x, cur_y] = intensity_arr
            # when at right edge, direction of next step in x should be backward
            if cur_y == self.y - 1 and shift != -1:
                cur_x -= 1 # move up one row
                shift *= -1 # flip direction
            # when at left edge, direction of next step in x should be forward
            elif cur_y == 0 and shift != 1:
                cur_x -= 1 # move up one row
                shift *= -1 # flip direction
            else: # when not at edge, step in x
                cur_y += shift # move to next column

        raman_image = rp.SpectralImage(spectral_data, spectral_axis)

        return raman_image
    
    def raman_plot(self, band):
        rp.plot.image(self.raman_image.band(band))
        rp.plot.show()
        return

    def get_integrals(self, slices=100):
        path = Path(self.path) # store as Path object for easier manipulation
        files = list(path.glob('*.txt')) # extract .txt files and store as list

        # if no files found
        if not files:
            print("No .txt files found in that directory.")
            return 
        
        # sort the files in the order of their file number
        files.sort(key=self.get_file_number)

        # read one file, only the raman_shift column, choose that raman_shift column and convert tolist
        #read from only one file (since raman_shift shared across all files)
        raman_shifts = pd.read_csv(
            files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0]
            )['raman_shift'].tolist()

        # intensities, raman shifts (might not need this, depends on future applications)
        #spectral_data = np.zeros(shape=(self.x, self.y, len(raman_shifts))) # matrix of x, y, each element holding a spectra
        #spectral_axis = np.array(raman_shifts)

        # list to hold area under curve
        integrals = np.zeros(shape=(self.x, self.y))
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
            intensity_arr = pd.read_csv(
                file, sep='\t', names=['intensity'], header=None,usecols=[1],
                )['intensity'].tolist()
            
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

            # assign integral to its grid position
            integrals[cur_x, cur_y] = integral
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

        return intensity_slice, integral_slice, integrals, raman_shifts
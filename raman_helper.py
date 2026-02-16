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
    
    def get_area(self, pipeline):
        
        # get files
        files = self.get_files()

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()
        
        # list to hold area under curve
        integrals = np.zeros(shape=(self.x, self.y))

        if pipeline == 1:
            pipeline = rp.preprocessing.protocols.georgiev2023_P1(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 2:
            pipeline = rp.preprocessing.protocols.georgiev2023_P2(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 3:
            pipeline = rp.preprocessing.protocols.georgiev2023_P3(normalisation_pixelwise=True, fingerprint=False)

        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)

        for file in files:
            # read the intensity column and store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None, usecols=[1],)['intensity'].tolist()
            
            # make raman spectra object and preprocess spectra
            raman_spectra = rp.Spectrum(intensity_arr, raman_shifts)
            raman_spectra = pipeline.apply(raman_spectra)

            # get area under whole curve
            integral = np.trapezoid(intensity_arr, raman_shifts)

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
    
    def preprocess(self):
        
        # despike suitable if using ccd
        return

    def apply_preprocessing(self, cell_layer):
        # this pipeline was designed for cell / tissue maps
        preprocessing_pipeline = rp.preprocessing.Pipeline([
            #rp.preprocessing.misc.Cropper(region=(500, 1800)),
            rp.preprocessing.despike.WhitakerHayes(),
            rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
            rp.preprocessing.baseline.ASLS(),
            rp.preprocessing.normalise.MinMax(pixelwise=False),
        ])
        return preprocessing_pipeline.apply(cell_layer) 

    def get_area_regions(self, pipeline, regions):
        """
        Function to find the area under the curve by regions.
        
        Returns:
        area_by_region - integral of region, stored as: list, shape=(x, y, len(spectra)) 
        shift_by_region - raman shift stored as dictionary: {region : spectra of region} (same for all pixels)
        spectra_by_region - preprocessed spectra of region, stored as: { pixel, region : spectra of pixel and region}
        original_spectra_by_region - original spectra of region, stored as: { pixel, region : spectra of pixel and region}

        original_spectra_by_region was added to compare with preprocessed data

        :param pipeline: pipeline to use. Currently using predefined pipelines from RamanSPy library.
        :param regions: Number of regions to split the spectra into.
        """
        # get files
        files = self.get_files()

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()
        
        div = len(raman_shifts) // regions
        area_by_region = np.zeros(shape=(self.x, self.y, regions))

        # key: pixel, region. value: spectra of that region for that pixel
        spectra_by_region = {}
        original_spectra_by_region = {}
        
        # key: region. value: spectra of that region
        shift_by_region = {}

        # store raman shift by region
        i, j, region = 0, div - 1, 0
        #TODO: fix this, the last term not included when i:j
        while i < len(raman_shifts):
            shift_by_region[region] = [f"{raman_shifts[i]} - {raman_shifts[j]}", raman_shifts[i:j]]
            region += 1
            i += div
            j += div
        
        if pipeline == 1:
            pipeline = rp.preprocessing.protocols.georgiev2023_P1(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 2:
            pipeline = rp.preprocessing.protocols.georgiev2023_P2(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 3:
            pipeline = rp.preprocessing.protocols.georgiev2023_P3(normalisation_pixelwise=True, fingerprint=False)

        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)
        pixel = 1
        for file in files:
            # read the intensity column and store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None, usecols=[1],)['intensity'].tolist()
            
            # make raman spectra object and preprocess spectra
            raman_spectra = rp.Spectrum(intensity_arr, raman_shifts)
            raman_spectra = pipeline.apply(raman_spectra)

            # process data
            processed_data = raman_spectra.spectral_data

            # pointers for splitting
            i, j, region = 0, div - 1, 0

            # store data by regions
            while i < len(raman_shifts):
                # Note that does eg. 0 133 and 134 267, so relation between 133 and 134 are is missed
                area = np.trapezoid(processed_data[i:j], raman_shifts[i:j])
                # store area by region
                area_by_region[cur_x, cur_y, region] = area
                # store spectras
                spectra_by_region[pixel, region] = processed_data[i:j]
                original_spectra_by_region[pixel, region] = intensity_arr[i:j] # store original for comparison
                # update pointers
                region += 1
                i += div
                j += div

            # grid stepping logic
            if cur_y == self.y - 1 and step != -1: # on right boundary
                cur_x -= 1 # step up
                step *= -1 # flip step direction
            elif cur_y == 0 and step != 1: # on left boundary
                cur_x -= 1 # step up
                step *= -1 # flip step direction
            else: # not on boundary
                cur_y += step # step 
            pixel += 1
        return area_by_region, shift_by_region, spectra_by_region, original_spectra_by_region

    def get_all_slice(self):
        #TODO: use flat? use mean?
        # get files
        files = self.get_files()

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()
        raman_slices = []

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
        
        # store slice
        raman_slices.append(rp.SpectralContainer(spectral_data, spectral_axis))

        # return all slices and raman_shift
        return raman_slices, spectral_axis
    
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
    
    def get_slice_pipeline(self, pipeline):
        # get files
        files = self.get_files()

        # read in raman_shifts and store as list (only from one file, since same in all files)
        raman_shifts = pd.read_csv(files[0], sep='\t', names=['raman_shift'], header=None, usecols=[0])['raman_shift'].tolist()

        spectral_data = np.zeros(shape=(self.x, self.y, len(raman_shifts))) 
        spectral_axis =  np.array(raman_shifts)
        
        # position of grid 1 at bottom-left corner
        cur_x, cur_y =  self.x - 1, 0
        step = 1 # intially steps forward (right)
  
        for file in files:
            # read the intensity column, store as list
            intensity_arr = pd.read_csv(file, sep='\t', names=['intensity'], header=None,usecols=[1],)['intensity'].tolist()
            
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

        if pipeline == 1:
            pipeline = rp.preprocessing.protocols.georgiev2023_P1(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 2:
            pipeline = rp.preprocessing.protocols.georgiev2023_P2(normalisation_pixelwise=True, fingerprint=False)
        elif pipeline == 3:
            pipeline = rp.preprocessing.protocols.georgiev2023_P3(normalisation_pixelwise=True, fingerprint=False)
        

        raman_slice = rp.SpectralContainer(spectral_data, spectral_axis)
        raman_slice = pipeline.apply(raman_slice)
        return raman_slice
    
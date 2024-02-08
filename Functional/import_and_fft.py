import numpy as np
import NSFopen as ns
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy.fft as fft
from mpl_toolkits.axes_grid1 import make_axes_locatable

current_dir = Path(os.getcwd())
parent_dir = current_dir.parent
data_folder = "/Users/thomashartigan/Library/CloudStorage/GoogleDrive-tjh200@cam.ac.uk/My Drive/E2/Images/Initial Testing/Setup B/" #parent_dir.as_posix() +'/Raw_Data/'

class Data_Image():
    def __init__(self, image_number):
        data_folder = "/Users/thomashartigan/Library/CloudStorage/GoogleDrive-tjh200@cam.ac.uk/My Drive/E2/Images/Initial Testing/Setup B/"
        self.image_path = data_folder + "Image" + image_number + ".nid"
        self.import_data()
    
    def import_data(self):
        afm = ns.read(self.image_path)
        data = afm.data # raw data
        param = afm.param # parameters
        self.params = param
        self.size = self.params['Scan']['range']['Value'][0]
        #print(data.keys())
        #print(param.keys())
        self.forward_current = data['Image']['Forward']['Tip Current']
        self.forward_current_frequencies, self.forward_current_fft = self.fft_data(self.forward_current)
        self.forward_current_peaks_array, self.forward_current_peak_image = self.detect_peaks(np.log10(abs(self.forward_current_fft)))
        self.backward_current = data['Image']['Backward']['Tip Current']
        self.backward_current_frequencies, self.backward_current_fft = self.fft_data(self.backward_current)
        self.backward_current_peaks_array, self.backward_current_peak_image = self.detect_peaks(np.log10(abs(self.backward_current_fft)))
        self.forward_z = data['Image']['Forward']['Z-Axis']
        self.forward_z_frequencies, self.forward_z_fft = self.fft_data(self.forward_z)
        self.forward_z_peaks_array, self.forward_z_peak_image = self.detect_peaks(np.log10(abs(self.forward_z_fft)))
        self.backward_z = data['Image']['Backward']['Z-Axis']
        self.backward_z_frequencies, self.backward_z_fft = self.fft_data(self.backward_z)
        self.backward_z_peaks_array, self.backward_z_peak_image = self.detect_peaks(np.log10(abs(self.backward_z_fft)))
        
        print(self.size)

    def fft_data(self, dataset):
        dataset_fft = fft.fft2(dataset)
        n = len(dataset[0]) # Assuming a square dataset
        step_length = self.size/n
        freq = np.fft.fftfreq(n, d = step_length)
        shifted_freqs = fft.fftshift(dataset_fft)
        cleaned_shifted_freqs = self.clean_fft_data(shifted_freqs)
        return(freq, cleaned_shifted_freqs)
        
    def clean_fft_data(self, shifted_fft_data):
        #Remove 0 frequency component (should use even numbers of datapoints in rows)
        n = len(shifted_fft_data[0])
        target_n = int(n/2)
        shifted_fft_data[target_n][target_n] = 1E-10
        #shifted_fft_data[np.log10(abs(shifted_fft_data)) < -6.8] = 1E-
        return shifted_fft_data


    def plot_image_and_ffts(self):
        real_space_length = self.size
        im = [[None, None, None, None],[None, None, None, None],[None, None, None, None]]

        fig, axs = plt.subplots(3, 4, figsize=(15,9))
        # Height images
        im[0][0] = axs[0,0].imshow(self.forward_current, interpolation = 'none', extent=[0, real_space_length, 0, real_space_length], aspect=1)
        axs[0,0].set_title('Forward Current')

        im[0][1] = axs[0,1].imshow(self.backward_current, interpolation = 'none', extent=[0, real_space_length, 0, real_space_length], aspect=1)
        axs[0,1].set_title('Backward Current')

        im[0][2] = axs[0,2].imshow(self.forward_z, interpolation = 'none', extent=[0, real_space_length, 0, real_space_length], aspect=1)
        axs[0,2].set_title('Forward Z')

        im[0][3] = axs[0,3].imshow(self.backward_z, interpolation = 'none', extent=[0, real_space_length, 0, real_space_length], aspect=1)
        axs[0,3].set_title('Backward Z')

        # Fourier transform images
        im[1][0] = axs[1,0].imshow(np.log10(abs(self.forward_current_fft)), interpolation = 'none', extent=[min(self.forward_current_frequencies), max(self.forward_current_frequencies), min(self.forward_current_frequencies), max(self.forward_current_frequencies)], aspect=1)
        axs[1,0].set_title('Forward Current FFT')

        im[1][1] = axs[1,1].imshow(np.log10(abs(self.backward_current_fft)), interpolation = 'none', extent=[min(self.backward_current_frequencies), max(self.backward_current_frequencies), min(self.backward_current_frequencies), max(self.backward_current_frequencies)], aspect=1)
        axs[1,1].set_title('Backward Current FFT')

        im[1][2] = axs[1,2].imshow(np.log10(abs(self.forward_z_fft)), interpolation = 'none', extent=[min(self.forward_z_frequencies), max(self.forward_z_frequencies), min(self.forward_z_frequencies), max(self.forward_z_frequencies)], aspect=1)
        axs[1,2].set_title('Forward Z FFT')

        im[1][3] = axs[1,3].imshow(np.log10(abs(self.backward_z_fft)), interpolation = 'none', extent=[min(self.backward_z_frequencies), max(self.backward_z_frequencies), min(self.backward_z_frequencies), max(self.backward_z_frequencies)], aspect=1)
        axs[1,3].set_title('Backward Z FFT')

        print(self.forward_current_peak_image)
        im[2][0] = axs[2,0].imshow(self.forward_current_peak_image, interpolation = 'none', extent=[min(self.forward_current_frequencies), max(self.forward_current_frequencies), min(self.forward_current_frequencies), max(self.forward_current_frequencies)], aspect=1)
        axs[2,0].set_title('Forward Current Peaks')

        im[2][1] = axs[2,1].imshow(self.backward_current_peak_image, interpolation = 'none', extent=[min(self.backward_current_frequencies), max(self.backward_current_frequencies), min(self.backward_current_frequencies), max(self.backward_current_frequencies)], aspect=1)
        axs[2,1].set_title('Backward Current Peaks')

        im[2][2] = axs[2,2].imshow(self.forward_z_peak_image, interpolation = 'none', extent=[min(self.forward_z_frequencies), max(self.forward_z_frequencies), min(self.forward_z_frequencies), max(self.forward_z_frequencies)], aspect=1)
        axs[2,2].set_title('Forward Z Peaks')

        im[2][3] = axs[2,3].imshow(self.backward_z_peak_image, interpolation = 'none', extent=[min(self.backward_z_frequencies), max(self.backward_z_frequencies), min(self.backward_z_frequencies), max(self.backward_z_frequencies)], aspect=1)
        axs[2,3].set_title('Backward Z Peaks')
        # Peak images
        for i in range(0,3):
            for j in range(0,4):
                divider = make_axes_locatable(axs[i,j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im[i][j], cax=cax, orientation='vertical')
        fig.tight_layout()
        return(fig, axs)

    def detect_peaks(self, imagefft_dataset):
        """Should be used with logarithmic datasets"""
        original_imagefft_dataset = np.array(imagefft_dataset)
        print(original_imagefft_dataset)
        max_inds = []
        max_inds.append(np.array([i for i in np.unravel_index(np.argmax(imagefft_dataset, axis=None), imagefft_dataset.shape)]))
        imagefft_dataset[max_inds[0][0]][max_inds[0][1]] = -10
        while len(max_inds) < 6:
            ind = np.array(np.unravel_index(np.argmax(imagefft_dataset, axis=None), imagefft_dataset.shape))
            skip = False
            for max_ind in max_inds:
                if np.linalg.norm(ind-max_ind) < 4:
                    #print(f"Discarding maximum at {max_ind}")
                    imagefft_dataset[ind[0]][ind[1]] = -10
                    skip = True
            if skip == False:
                max_inds.append(ind)
                imagefft_dataset[ind[0]][ind[1]] = -10
        print(original_imagefft_dataset)
        peak_image = np.zeros(np.shape(imagefft_dataset))
        
        for max_ind in max_inds:
            peak_image[max_ind[0]][max_ind[1]]=original_imagefft_dataset[max_ind[0]][max_ind[1]]
            print(original_imagefft_dataset[max_ind[0]][max_ind[1]])

        print(original_imagefft_dataset)
        peak_image[peak_image == 0] = np.min(peak_image, axis=None) - 1
        print(np.min(peak_image, axis=None))
        print(peak_image[56][57])
        return max_inds, peak_image 
    
#A04782 B14000 #B14005 #A05424
image_of_interest = Data_Image("14000")
fig, axs = image_of_interest.plot_image_and_ffts()
fig.show()
plt.show()
#plt.imshow(image_of_interest.detect_peaks(np.log10(abs(image_of_interest.forward_current_fft))), cmap = 'Blues')
#plt.show()
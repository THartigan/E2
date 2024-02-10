import numpy as np
import NSFopen as ns
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy.fft as fft
from mpl_toolkits.axes_grid1 import make_axes_locatable

current_dir = Path(os.getcwd())
parent_dir = current_dir.parent
data_folder = "/Users/thomashartigan/Library/CloudStorage/GoogleDrive-tjh200@cam.ac.uk/My Drive/E2/Images/Initial Testing/Setup A/" #parent_dir.as_posix() +'/Raw_Data/'

class Data_Image():
    def __init__(self, image_number):
        data_folder = "/Users/thomashartigan/Library/CloudStorage/GoogleDrive-tjh200@cam.ac.uk/My Drive/E2/Images/Initial Testing/Setup A/"
        self.image_path = data_folder + "Image" + image_number + ".nid"
        self.import_data()
    
    def import_data(self):
        afm = ns.read(self.image_path)
        data = afm.data # raw data
        param = afm.param # parameters
        self.params = param
        self.peak_inverse_distances = []
        self.lattice_param_estimates = []
        self.size = self.params['Scan']['range']['Value'][0]
        #print(data.keys())
        #print(param.keys())
        self.forward_current = data['Image']['Forward']['Tip Current']
        self.forward_current_frequencies, self.forward_current_fft = self.fft_data(self.forward_current, True)
        self.forward_current_peaks_array, self.forward_current_peak_image = self.detect_peaks(np.log10(abs(self.forward_current_fft)))
        self.backward_current = data['Image']['Backward']['Tip Current']
        self.backward_current_frequencies, self.backward_current_fft = self.fft_data(self.backward_current, True)
        self.backward_current_peaks_array, self.backward_current_peak_image = self.detect_peaks(np.log10(abs(self.backward_current_fft)))
        self.forward_z = data['Image']['Forward']['Z-Axis']
        #self.forward_z = self.rotate_z_data(self.forward_z)
        self.forward_z_frequencies, self.forward_z_fft = self.fft_data(self.forward_z, False)
        #self.forward_z = np.abs(fft.ifft2(self.forward_z_fft))
        self.forward_z_peaks_array, self.forward_z_peak_image = self.detect_peaks(np.log10(abs(self.forward_z_fft)))
        self.backward_z = data['Image']['Backward']['Z-Axis']
        self.backward_z = self.rotate_z_data(self.backward_z)
        self.backward_z_frequencies, self.backward_z_fft = self.fft_data(self.backward_z, False)
        self.backward_z_peaks_array, self.backward_z_peak_image = self.detect_peaks(np.log10(abs(self.backward_z_fft)))
        
        
    def rotate_z_data(self, dataset):
        x_angle = self.params['Scan']['rotation']['Value'][1]
        y_angle = self.params['Scan']['rotation']['Value'][0]
        n = len(dataset[0])
        step_length = self.size / n
        x_corrections = np.linspace(0, self.size, n) * np.tan(x_angle)
        y_corrections = np.transpose([np.linspace(self.size, 0, n) * np.tan(y_angle)])
        xy_corrections = x_corrections + y_corrections
        print(xy_corrections) 
        return xy_corrections/1000 +dataset
        #X_corrections, Y_corrections = np.meshgrid(x_corrections + y_corrections)
        

        

    def fft_data(self, dataset, is_current: bool):
        dataset_fft = fft.fft2(dataset)
        #dataset_fft[0] = 0
        cleaned_fft = self.clean_fft_data(dataset_fft, is_current)
        n = len(dataset[0]) # Assuming a square dataset
        step_length = self.size/n
        freq = np.fft.fftfreq(n, d = step_length)
        shifted_fft = fft.fftshift(cleaned_fft)
        
        return(freq, shifted_fft)
        
    def clean_fft_data(self, fft_data, is_current : bool):
        #Remove 0 frequency component (should use even numbers of datapoints in rows)
        fft_data[0] = 0
        fft_data[:,0] = 0

        n = len(fft_data[0])
        self.middle_n = int(n/2)
        #shifted_fft_data[self.middle_n][self.middle_n] = 1E-10
        #shifted_fft_data[np.log10(abs(shifted_fft_data)) < -6.8] = 1E-
        return fft_data


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

        im[2][0] = axs[2,0].imshow(self.forward_current_peak_image, interpolation = 'none', extent=[min(self.forward_current_frequencies), max(self.forward_current_frequencies), min(self.forward_current_frequencies), max(self.forward_current_frequencies)], aspect=1, cmap = 'winter')
        axs[2,0].set_title('Forward Current Peaks')
        axs[2,0].set_facecolor('xkcd:salmon')

        im[2][1] = axs[2,1].imshow(self.backward_current_peak_image, interpolation = 'none', extent=[min(self.backward_current_frequencies), max(self.backward_current_frequencies), min(self.backward_current_frequencies), max(self.backward_current_frequencies)], aspect=1, cmap = 'winter')
        axs[2,1].set_title('Backward Current Peaks')
        axs[2,1].set_facecolor('xkcd:salmon')

        im[2][2] = axs[2,2].imshow(self.forward_z_peak_image, interpolation = 'none', extent=[min(self.forward_z_frequencies), max(self.forward_z_frequencies), min(self.forward_z_frequencies), max(self.forward_z_frequencies)], aspect=1, cmap = 'winter')
        axs[2,2].set_title('Forward Z Peaks')
        axs[2,2].set_facecolor('xkcd:salmon')

        im[2][3] = axs[2,3].imshow(self.backward_z_peak_image, interpolation = 'none', extent=[min(self.backward_z_frequencies), max(self.backward_z_frequencies), min(self.backward_z_frequencies), max(self.backward_z_frequencies)], aspect=1, cmap = 'winter')
        axs[2,3].set_title('Backward Z Peaks')
        axs[2,3].set_facecolor('xkcd:salmon')
        # Peak images
        for i in range(0,3):
            for j in range(0,4):
                divider = make_axes_locatable(axs[i,j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im[i][j], cax=cax, orientation='vertical')
        fig.tight_layout()
        return(fig, axs)
    
    def fourier_distance(self, pixel_length):
        """Converts the number of pixels to the corresponding inverse distance in the fourier domain"""
        length_per_pixel = max(self.forward_current_frequencies) * 2 / len(self.forward_current_fft[0])
        return np.array(pixel_length) * length_per_pixel
    
    def real_distance(self, fourier_pixel_length):
        return 1/self.fourier_distance(fourier_pixel_length)




    def detect_peaks(self, imagefft_dataset):
        """Should be used with logarithmic datasets"""
        original_imagefft_dataset = np.array(imagefft_dataset)
        max_inds = []
        max_inds.append(np.array([i for i in np.unravel_index(np.argmax(imagefft_dataset, axis=None), imagefft_dataset.shape)]))
        imagefft_dataset[max_inds[0][0]][max_inds[0][1]] = -np.infty
        attempts = 0
        while len(max_inds) < 6 and attempts < 1000:
            
            ind = np.array(np.unravel_index(np.argmax(imagefft_dataset, axis=None), imagefft_dataset.shape))
            skip = False
            for max_ind in max_inds:
                if self.fourier_distance(np.linalg.norm(ind-max_ind)) < 1/0.4E-9 or self.fourier_distance(np.linalg.norm(ind-[self.middle_n, self.middle_n])) > 1/0.15E-9 or abs(original_imagefft_dataset[ind[0]][ind[1]] - original_imagefft_dataset[max_inds[0][0]][max_inds[0][1]]) > 1:
                    attempts += 1
                    #print(f"Discarding maximum at {max_ind}")
                    #print(np.linalg.norm(ind))
                    #print(attempts)
                    imagefft_dataset[ind[0]][ind[1]] = -np.infty
                    skip = True
            if skip == False:
                max_inds.append(ind)
                imagefft_dataset[ind[0]][ind[1]] = -np.infty

        peak_image = np.zeros(np.shape(imagefft_dataset))
        
        for max_ind in max_inds:
            peak_image[max_ind[0]][max_ind[1]]=original_imagefft_dataset[max_ind[0]][max_ind[1]]

        peak_image[peak_image == 0] = -np.infty

        self.peak_displacements(max_inds)
        return max_inds, peak_image 
    
    def peak_displacements(self, max_inds):
        pixel_displacements = np.array(max_inds) - np.array([self.middle_n, self.middle_n])

        distance_displacements = self.fourier_distance([np.linalg.norm(vector) for vector in pixel_displacements])
        average_displacement = np.average(1/distance_displacements * 2/3)
        if average_displacement < 0.16E-9 and average_displacement > 0.12E-9:
            self.peak_inverse_distances.append(distance_displacements)
            self.lattice_param_estimates.append(average_displacement)
        else:
            print("lattice param was thought unreasonable")

    
#A04782 B14000 #B14005 #A05424
image_of_interest = Data_Image("04790")
fig, axs = image_of_interest.plot_image_and_ffts()
print(image_of_interest.lattice_param_estimates)
fig.show()
plt.show()
#plt.imshow(image_of_interest.detect_peaks(np.log10(abs(image_of_interest.forward_current_fft))), cmap = 'Blues')
#plt.show()
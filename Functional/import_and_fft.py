import numpy as np
import NSFopen as ns
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy.fft as fft
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings("ignore", message="divide by zero encountered in log10")

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
        #self.backward_z = self.rotate_z_data(self.backward_z)
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
        #print(xy_corrections) 
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
        #print(len(self.forward_current_fft[0]-1))
        length_per_pixel = (max(self.forward_current_frequencies) - min(self.forward_current_frequencies)) / (len(self.forward_current_fft[0])-1)
        return np.array(pixel_length) * length_per_pixel
    
    def real_distance(self, fourier_pixel_length):
        return 1/self.fourier_distance(fourier_pixel_length)
    
    def fourier_pixel_vectors_to_real_vector(self, vectors):
        real_vectors = []
        for vector in vectors:
            vector = np.array(vector)
            norm = np.linalg.norm(vector)
            length_per_pixel = (max(self.forward_current_frequencies) - min(self.forward_current_frequencies)) / (len(self.forward_current_fft[0])-1)
            #print(f'length per pixel: {length_per_pixel}')
            real_length = 1/(norm * length_per_pixel)
            #print(real_length)
            real_vector = real_length / (norm * length_per_pixel) * vector * length_per_pixel
            real_vectors.append(real_vector)
        return real_vectors




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

        # Find the peak displacements from the origin
        self.peak_displacements(max_inds)
        return max_inds, peak_image 
    
    def peak_displacements(self, max_inds):
        pixel_displacements = np.array(max_inds) - np.array([self.middle_n, self.middle_n])
        #[1,2]-> [2,-1], [-1,2] -> [2,1], [1,-2] -> [-2,-1] & [-1,-2] -> [-2,1]
        # none -> 2, 1 -> 1, 2 -> 1 and 12 -> 2
        pixel_displacements_cartesian = []
        for pixel_displacement in pixel_displacements:
            if pixel_displacement[0] <= 0 and pixel_displacement[1] <= 0:
                pixel_displacements_cartesian.append([pixel_displacement[1], -pixel_displacement[0]])
            elif pixel_displacement[0] <= 0:
                pixel_displacements_cartesian.append([pixel_displacement[1], -pixel_displacement[0]])
            elif pixel_displacement[1] <= 0:
                pixel_displacements_cartesian.append([pixel_displacement[1], -pixel_displacement[0]])
            else:
                pixel_displacements_cartesian.append([pixel_displacement[1], -pixel_displacement[0]])
            #print(f'{pixel_displacement} -> {pixel_displacements_cartesian[len(pixel_displacements_cartesian)-1]}')
        
        ls = self.get_ls(pixel_displacements_cartesian)
        params_init = [2.15E-10, np.pi /6, np.pi /6, np.pi /6]
        #print(l0, l1)
        if len(ls) > 1:
            params = fsolve(self.rotation_equations, x0=params_init, args=ls)
            ##print(params)
            rotated_ls = []
            for l in ls:
                rotated_ls.append(self.unrotate([l[0], l[1], 0], -params[1], -params[2], -params[3]))
            ##print(rotated_ls)
        else:
            print("insufficient points to find rotations")
        
        #print(pixel_displacements_cartesian)
        
        #distance_displacements = self.fourier_distance([np.linalg.norm(vector) for vector in pixel_displacements_cartesian])
        #print(distance_displacements)
        print(f"lattice param estimate: {params[0] * 2/3}")
        #average_displacement = np.average(1/distance_displacements * 2/3)
        #if average_displacement < 0.16E-9 and average_displacement > 0.12E-9:
        #    self.peak_inverse_distances.append(distance_displacements)
        #    self.lattice_param_estimates.append(average_displacement)
        #else:
        #    print("lattice param was thought unreasonable")

    def get_ls(self, peak_vectors):
        ls = []
        #print(f"peak vectors: {peak_vectors}")
        peak_vectors = np.array(peak_vectors)
        for i in range(0,3):
                if len(peak_vectors) > 0:
                    #print(f"peak vectors: {peak_vectors}")
                    peak_vectors = peak_vectors[peak_vectors[:,1]>0]
                    # Finds the points furthest in the positive x-direction
                    l_draws = peak_vectors[(np.argwhere(peak_vectors[:,0] == np.max(peak_vectors, axis=0)[0])).flatten()]
                    # If there is such a point, then find the one of these closest to the x-axis
                    if len(l_draws) != 0:
                        l_win = l_draws[(np.argwhere(l_draws[:,1] == np.min(l_draws, axis=0)[1])).flatten()].flatten()
                        if np.linalg.norm(self.fourier_pixel_vectors_to_real_vector(l_win)) > 1E-10:
                            pv_list = peak_vectors.tolist()
                            pv_list.remove(list(l_win))
                            peak_vectors = np.array(pv_list)
                            ls.append(l_win.tolist())
        #if np.arccos(np.dot(l0_and_l1[0], l0_and_l1[1]) / (np.linalg.norm(l0_and_l1[0]) * np.linalg.norm(l0_and_l1[1]))) > np.pi * 70/180:
        #    print("Actually dealing with l2")
        #print(f'frequency coords: {l0_and_l1}')
        ls = self.fourier_pixel_vectors_to_real_vector(ls)
        ##print(f'real space positions: {ls}')
        return (ls)
    
    def rotation_equations(self, w, ls):
        # w = (l0, alpha, beta, gamma)
        
        l0 = w[0]
        alpha = w[1]
        beta = w[2]
        gamma = w[3]
        
        if len(ls) == 3:
            l1_est = self.rotate([l0, 0, 0], alpha, beta, gamma)
            l2_est = self.rotate([l0 * np.cos(np.pi/3), l0 * np.sin(np.pi/3), 0], alpha, beta, gamma)
            l3_est = self.rotate([-l0 * np.cos(np.pi/3), l0 * np.sin(np.pi/3), 0], alpha, beta, gamma)
        
            l1x = ls[0][0]
            l1y = ls[0][1]
            l2x = ls[1][0]
            l2y = ls[1][1]
            l3x = ls[2][0]
            l3y = ls[2][1]
            
            F = np.zeros(4)
            F[0] = l2_est[0]-l3_est[0] - (l2x-l3x)
            F[1] = l2_est[1]-l3_est[1] - (l2y-l3y)
            F[2] = l1_est[0]-l2_est[0] - (l1x-l2x)
            F[3] = l1_est[1]-l2_est[1] - (l1y-l2y)
        else:
            F = [0,0,0,0]
            print("not enough ls")
            
        #F[0] = l1_est[0] - l1x
        #print(l1_est[0], l1x)
        #F[1] = l1_est[1] - l1y
        #F[2] = l2_est[0] - l2x
        #F[3] = l2_est[1] - l2y
        #if len(ls) == 3:
        #    F[4] = l3_est[0] - l3x
        #    F[5] = l3_est[1] - l3y
        return F
    
    def rotate(self, vector, alpha, beta, gamma):
        cos = np.cos
        sin = np.sin
        Rx = np.matrix([[1, 0, 0],
                        [0, cos(alpha), -sin(alpha)],
                        [0, sin(alpha), cos(alpha)]])
        Ry = np.matrix([[cos(beta), 0, sin(beta)],
                        [0, 1, 0],
                        [-sin(beta), 0, cos(beta)]])
        Rz = np.matrix([[cos(gamma), -sin(gamma), 0],
                        [sin(gamma), cos(gamma), 0],
                        [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        #R = np.matrix([[cos(beta) * cos(gamma), sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma), cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)],
         #              [cos(beta) * sin(gamma), sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma), cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)],
         #              [-sin(beta), sin(alpha) * cos(beta), cos(alpha) * cos(beta)]])
        #print(np.matmul(R, np.transpose(np.matrix(vector))).flatten().tolist()[0])
    
        return np.matmul(R, np.transpose(np.matrix(vector))).flatten().tolist()[0]

    def unrotate(self, vector, alpha, beta, gamma):
        cos = np.cos
        sin = np.sin
        Rx = np.matrix([[1, 0, 0],
                        [0, cos(alpha), -sin(alpha)],
                        [0, sin(alpha), cos(alpha)]])
        Ry = np.matrix([[cos(beta), 0, sin(beta)],
                        [0, 1, 0],
                        [-sin(beta), 0, cos(beta)]])
        Rz = np.matrix([[cos(gamma), -sin(gamma), 0],
                        [sin(gamma), cos(gamma), 0],
                        [0, 0, 1]])
        R = Rx @ Ry @ Rz
        return np.squeeze(np.asarray(R @ np.transpose(np.matrix(vector))))
#A04782 B14000 #B14005 #A05424 #A04790 #B14075 #A07050
image_of_interest = Data_Image("04790")
a = image_of_interest.rotate([1,1,1], 0.3, 0.2, 1.5)
##print(image_of_interest.unrotate(a, -0.3, -0.2, -1.5))
fig, axs = image_of_interest.plot_image_and_ffts()
#print(image_of_interest.lattice_param_estimates)
plt.show()
fig.show()


#plt.imshow(image_of_interest.detect_peaks(np.log10(abs(image_of_interest.forward_current_fft))), cmap = 'Blues')
#plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class resizor():
    def __init__(self, image_path, downscale_factor, filter_side, sigma):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.downsampled_image = self.resize(1/downscale_factor)
        self.image_distribution = self.determine_distribution()
        self.plainly_upscaled_image = self.image * 0
        self.plain_upscale(downscale_factor)
        self.gaussian_filter = self.make_gaussian_filter(filter_side, sigma)
        
        self.updated_upscaled_image = self.plainly_upscaled_image.copy()
        self.updated_boolean_matrix = np.zeros(self.image.shape) # 1 if pixel i,j has been updated, 0 otherwise
        self.updated_image_distribution = self.image_distribution.copy()

    def resize(self, factor):
        return cv2.resize(self.image, (0, 0), fx = factor, fy = factor)
    
    def determine_distribution(self):
        distribution = np.zeros(256, dtype=np.int64)
        for i in range(256):
            distribution[i] = len(np.where(self.image == i)[0])
        return distribution
    
    def plain_upscale(self, downscale_factor):
        h, w = self.downsampled_image.shape
        for i in range(h):
            for j in range(w):
                start_row, end_row = i * downscale_factor, (i + 1) * downscale_factor
                start_col, end_col = j * downscale_factor, (j + 1) * downscale_factor
                self.plainly_upscaled_image[start_row : end_row, start_col : end_col] = self.downsampled_image[i, j]

    def make_gaussian_filter(self, filter_side, sigma, plot=False):
        start, stop = -1 * (filter_side // 2), (filter_side // 2) + 1
        x_values = np.arange(start, stop, step=1)
        distribution = np.exp(-1 * (x_values ** 2) / (2 * sigma ** 2))
        normalized_distribution = distribution / np.sum(distribution)
        if plot:
            plt.plot(x_values, normalized_distribution)
            plt.show()
        normalized_distribution = np.reshape(normalized_distribution, (filter_side, 1))
        gaussian_filter = normalized_distribution @ normalized_distribution.T
        return gaussian_filter

    def convolution(self):
        filter_side = self.gaussian_filter.shape[0]
        conv_image = np.pad(self.updated_upscaled_image, filter_side)
        
        center_idx = filter_side // 2
        filter_center = self.gaussian_filter[center_idx, center_idx]
        no_center_filter = self.gaussian_filter.copy()
        no_center_filter[center_idx, center_idx] = 0

        plain_pixels_y = np.where(self.updated_boolean_matrix == 0)[0] # y coors of pixels that have not been updated (they were 'plainly' upscaled)
        plain_pixels_x = np.where(self.updated_boolean_matrix == 0)[1] # x coors of pixels that have not been updated (still 'plain') 
        plain_pixels_list = np.concat((np.expand_dims(plain_pixels_y, axis=1), np.expand_dims(plain_pixels_x, axis=1)), axis=1).tolist()

        chosen_pixels = np.ones(plain_pixels_y.shape, dtype=np.int64) * -1
        differences_from_targets = np.ones(plain_pixels_y.shape) * -1

        unchosen_pixels_y = plain_pixels_y.tolist()
        unchosen_pixels_x = plain_pixels_x.tolist()
        unchosen_count = len(unchosen_pixels_y)

        dummy_image_distribution = self.updated_image_distribution.copy()

        while unchosen_count > 0:
            possible_pixels = np.where(dummy_image_distribution > 0)[0]
            for idx in range(unchosen_count):
                i, j = unchosen_pixels_y[idx], unchosen_pixels_x[idx]
                start_row, end_row = i, i + filter_side
                start_col, end_col = j, j + filter_side
                ij_window = conv_image[start_row : end_row, start_col : end_col]
                window_average = np.sum(ij_window) / (filter_side ** 2)
                weighted_window_wo_ij = ij_window * no_center_filter # wo is short for without
                weighted_avg_wo_ij = np.sum(weighted_window_wo_ij)
                target_ij = (window_average - weighted_avg_wo_ij) / filter_center # window_average = weighted_avg
                abs_difference = np.abs(possible_pixels - target_ij)
                min_difference = np.min(abs_difference)
                closest_ijs = possible_pixels[np.where(abs_difference == min_difference)]
                chosen_ij = np.random.choice(closest_ijs)
                dummy_image_distribution[chosen_ij] -= 1

                plain_pixels_idx = plain_pixels_list.index([i, j])
                chosen_pixels[plain_pixels_idx] = chosen_ij
                differences_from_targets[plain_pixels_idx] = min_difference

            overchosen_pixels = np.where(dummy_image_distribution < 0)[0]
            unchosen_pixels_y = []
            unchosen_pixels_x = []

            for pixel in overchosen_pixels:
                chosen_pixel_indices = np.where(chosen_pixels == pixel)[0]
                chosen_pixel_differences = differences_from_targets[chosen_pixel_indices]
                sorted_differences_indices = np.argsort(chosen_pixel_differences)

                pixel_max_count = self.updated_image_distribution[pixel]

                difference_cutoff = chosen_pixel_differences[sorted_differences_indices[pixel_max_count - 1]]
                cutoff_pixel_indices = chosen_pixel_indices[np.where(chosen_pixel_differences <= difference_cutoff)]
                randomly_chosen_indices = np.random.choice(a=cutoff_pixel_indices, size=pixel_max_count, replace=False)
                unchosen_pixel_indices = np.array(list(set(chosen_pixel_indices) - set(randomly_chosen_indices)))

                dummy_image_distribution[pixel] += unchosen_pixel_indices.size
                if dummy_image_distribution[pixel] != 0:
                    print("PROBLEM:", pixel, "!= 0!")

                unchosen_pixels_y += plain_pixels_y[unchosen_pixel_indices].tolist()
                unchosen_pixels_x += plain_pixels_x[unchosen_pixel_indices].tolist()
                
            unchosen_count = len(unchosen_pixels_y)

        return chosen_pixels, differences_from_targets, plain_pixels_y, plain_pixels_x
    
    def conductor(self, percent_per_conv):
        w, h = self.image.shape
        pixel_count = w * h
        pixels_per_conv = int(pixel_count * percent_per_conv)
        conv_count = math.ceil(pixel_count / pixels_per_conv)

        for i in range(conv_count):
            print('Convolution', i)
            chosen_pixels, differences_from_targets, plain_pixels_y, plain_pixels_x = self.convolution()
            pixels_this_conv = min(pixels_per_conv, len(plain_pixels_y)) # the last convolution could have fewer pixels than pixels_per_conv
            sorted_difference_indices = np.argsort(differences_from_targets)

            difference_cutoff = differences_from_targets[sorted_difference_indices[pixels_this_conv - 1]]
            cutoff_pixels_indices = np.where(differences_from_targets <= difference_cutoff)[0]
            update_indices = np.random.choice(a=cutoff_pixels_indices, size=pixels_this_conv, replace=False)

            updated_pixels = chosen_pixels[update_indices]
            updated_pixels_coors = plain_pixels_y[update_indices], plain_pixels_x[update_indices]
            self.updated_upscaled_image[updated_pixels_coors] = updated_pixels
            self.updated_boolean_matrix[updated_pixels_coors] = 1

            updated_pixels_set = set(updated_pixels.tolist())
            for pixel in updated_pixels_set:
                pixel_count = np.where(updated_pixels == pixel)[0].size
                self.updated_image_distribution[pixel] -= pixel_count

r = resizor(image_path='downscaled.png', downscale_factor=2, filter_side=5, sigma=5)
r.conductor(percent_per_conv=0.1)
print(r.updated_image_distribution)

# cv2.imwrite('grayscale.jpg', r.image)
# cv2.imwrite('downscaled.png', r.downsampled_image)
# cv2.imwrite('plain_upscale.png', r.plainly_upscaled_image)


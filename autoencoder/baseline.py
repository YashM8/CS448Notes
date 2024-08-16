import numpy as np
import os

# Load data
data = []
for i in range(100):
    data.append(np.load(os.path.join('profiles', f'color{i}.npy')))

np.random.seed(1975)
np.random.shuffle(data)

# Split data into train and test sets
train_data = data[:90]
test_data = data[90:]

# Add noise to the data
train_data_noisy = [img.copy() for img in train_data]
test_data_noisy = [img.copy() for img in test_data]


for i in range(len(train_data_noisy)):
    train_data_noisy[i] += np.round(np.random.normal(0, 6, size=train_data_noisy[i].shape)).astype(int)

for i in range(len(test_data_noisy)):
    test_data_noisy[i] += np.round(np.random.normal(0, 6, size=test_data_noisy[i].shape)).astype(int)


# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
#
#
# # Define a function for wavelet denoising
# def wavelet_denoising(image, wavelet='db1', threshold=25):
#     # Perform wavelet transform
#     coeffs = pywt.wavedec2(image, wavelet)
#     coeffs_thresholded = list(coeffs)
#
#     # Apply threshold to detail coefficients
#     for i in range(1, len(coeffs_thresholded)):
#         coeffs_thresholded[i] = tuple(
#             pywt.threshold(c, threshold, mode='soft') for c in coeffs_thresholded[i]
#         )
#
#     # Reconstruct the image from the thresholded coefficients
#     denoised_image = pywt.waverec2(coeffs_thresholded, wavelet)
#
#     return np.clip(denoised_image, 0, 255)
#
#
# # Select the noisy image
# noisy_image = train_data_noisy[5]
#
# # Perform wavelet denoising
# denoised_image = wavelet_denoising(noisy_image, wavelet='db1', threshold=20)
#
# # Normalize images to the range [0, 1]
# noisy_image_normalized = noisy_image / 255.0
# denoised_image_normalized = denoised_image / 255.0
#
# # Plot before and after
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.title('Noisy Image')
# plt.imshow(noisy_image_normalized, cmap='gray')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title('Denoised Image')
# plt.imshow(denoised_image_normalized, cmap='gray')
# plt.axis('off')
#
# plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Select the noisy image
noisy_image = train_data_noisy[5]

# Convert to uint8 if necessary
if noisy_image.dtype != np.uint8:
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Perform Non-Local Means denoising using OpenCV
denoised_image = cv2.fastNlMeansDenoising(noisy_image, None, h=10, templateWindowSize=21, searchWindowSize=35)

# Normalize images to the range [0, 1]
noisy_image_normalized = noisy_image / 255.0
denoised_image_normalized = denoised_image / 255.0

# Plot before and after
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Noisy Image')
plt.imshow(noisy_image_normalized, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Denoised Image')
plt.imshow(denoised_image_normalized, cmap='gray')
plt.axis('off')

plt.show()

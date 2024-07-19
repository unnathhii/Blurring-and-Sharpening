import cv2
import numpy as np

# Function to remove noise from the image
def denoise_image(image):
    # Median filter (good for salt-and-pepper noise)
    denoised_median = cv2.medianBlur(image, 5)
    
    # Non-Local Means Denoising (good for gaussian noise)
    denoised_nlm = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21) 
    
    return denoised_median, denoised_nlm

# Reading the image
img = cv2.imread('noisy.jpg')

# Denoising the image
denoised_median, denoised_nlm = denoise_image(img)

# Gaussian blur for sharpening
gaussian_blur_median = cv2.GaussianBlur(denoised_median, (7, 7), 2)
gaussian_blur_nlm = cv2.GaussianBlur(denoised_nlm, (7, 7), 2)

# Sharpening using addWeighted()
sharpened_median1 = cv2.addWeighted(denoised_median, 2.5, gaussian_blur_median, -1.5, 0)
sharpened_median2 = cv2.addWeighted(denoised_median, 4.5, gaussian_blur_median, -3.5, 0)
sharpened_median3 = cv2.addWeighted(denoised_median, 7.5, gaussian_blur_median, -6.5, 0)

sharpened_nlm1 = cv2.addWeighted(denoised_nlm, 2.5, gaussian_blur_nlm, -1.5, 0)
sharpened_nlm2 = cv2.addWeighted(denoised_nlm, 4.5, gaussian_blur_nlm, -3.5, 0)
sharpened_nlm3 = cv2.addWeighted(denoised_nlm, 7.5, gaussian_blur_nlm, -6.5, 0)

# Showing the denoised and sharpened images
cv2.imshow('Sharpened Median 3', sharpened_median3)
cv2.imshow('Sharpened Median 2', sharpened_median2)
cv2.imshow('Sharpened Median 1', sharpened_median1)
cv2.imshow('Denoised Median', denoised_median)

cv2.imshow('Sharpened NLM 3', sharpened_nlm3)
cv2.imshow('Sharpened NLM 2', sharpened_nlm2)
cv2.imshow('Sharpened NLM 1', sharpened_nlm1)
cv2.imshow('Denoised NLM', denoised_nlm)

cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

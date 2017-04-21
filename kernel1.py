# import the necessary packages
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
def convolve(image, kernel):
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) / 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

	# loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
			k = (roi * kernel).sum()

			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k

	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	# return the output image
	return output




# Load an color image in grayscale
img = cv2.imread('sem_ic.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)





#cv2.imshow('image',img)
#cv2.waitKey(0) & 0xFF



#kernel 2

kernel1 = np.ones((3, 3), dtype="float") * (1.0 / (3 * 3))


Output = convolve(gray, kernel1)
convoleOutput1 = convolve(Output, kernel1)

#opencvOutput = cv2.filter2D(gray, -1, kernel2)

cv2.imshow('convoleOutput1',convoleOutput1)
cv2.waitKey(0) & 0xFF

#cv2.imshow('opencvOutput2',opencvOutput2)
#cv2.waitKey(0) & 0xFF

cv2.imwrite('convoledforkernel1_3x3.jpg',convoleOutput1)

cv2.destroyAllWindows()











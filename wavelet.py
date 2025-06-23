import numpy as np
import pywt
import cv2

def w2d(img, mode = 'haar', level = 1):
    imArray = img

    # convert to grayscale
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficient
    coeffs = pywt.wavedec2(imArray, mode, level = level)

    # Process coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0]*=0 # Delete LL (Low frequency)

    # Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H
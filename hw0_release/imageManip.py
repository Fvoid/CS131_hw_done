import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5 * np.square(image)
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale
    rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components:

    0.2989 * R + 0.5870 * G + 0.1140 * B
    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    out = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    ### END YOUR CODE
    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    out = image.copy()
    ### YOUR CODE HERE
    if channel == "R":
        out[...,0] = 0
    elif channel == "G":
        out[...,1] = 0
    else:
        out[...,2] = 0
    
    ### END YOUR CODE

    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = image.copy()

    ### YOUR CODE HERE
    if channel == "L":
        out[...,1] = 0
        out[...,2] = 0
    elif channel == "A":
        out[...,0] = 0
        out[...,2] = 0
    else:
        out[...,0] = 0
        out[...,1] = 0
    ### END YOUR CODE

    return out

def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = image.copy()

    ### YOUR CODE HERE
    if channel == "H":
        out[...,1] = 0
        out[...,2] = 0
    elif channel == "S":
        out[...,0] = 0
        out[...,2] = 0
    else:
        out[...,0] = 0
        out[...,1] = 0
    ### END YOUR CODE

    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    exclude = []
    image1c = image1.copy()
    image2c = image2.copy()
    ### YOUR CODE HERE
    for i in [channel1, channel2]:
        if i == "R":
            exclude.append(0)
        elif i == "G":
            exclude.append(1)
        else:
            exclude.append(2)
            
    image1c[..., exclude[0]] = 0
    image2c[..., exclude[1]] = 0
    width = image1c.shape[1]
    out = np.zeros_like(image1)
    out[:,:150,:] = image1c[:,:150,:]
    out[:, 150:301, :] = image2c[:, 150:301, :]
    ### END YOUR CODE

    return out

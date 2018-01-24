import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))


    #-1-kernel at the left top corner -2- right top corner -3- left bottom
    #-4- rihgt bottom -5- left side -6- right side -7-normal

    imagec = image.copy()
    imagec = zero_pad(imagec, 1, 1)
    hRange = int((Hk - 1) / 2)
    wRange = int((Wk - 1) / 2)
    ### YOUR CODE HERE
    for imageH in range(Hi):
        for imageW in range(Wi):
            sum1 = 0.0
            kerH = 0
            kerW = 0
            for covH in range(imageH-hRange, imageH+hRange+1):
                for covW in range(imageW-hRange, imageW+wRange+1):
                    if covH < Hi and covW < Wi and covH >= 0 and covW >= 0:
                        sum1 += image[covH, covW] * kernel[kerH, kerW]
                    kerW += 1
                    if kerW == Wk:
                        kerW = 0
                        kerH += 1
                    if kerH == Hk:
                        kerH =0
            out[imageH, imageW] = sum1

    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.pad(image, [(pad_height,), (pad_width, )], mode='constant')
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    padH = 0
    padW = 0

    imagec = image.copy()

    if Hk % 2 != 0:
        padH = int((Hk - 1) / 2)
    else:
        padH = int(Hk / 2)

    if Wk % 2 != 0:
        padW = int((Wk - 1) / 2)
    else:
        padW = int(Wk / 2)

    imagec = zero_pad(imagec, padH, padW)

    for imageH in range(Hi):
        for imageW in range(Wi):
            imagecH = imageH + padH
            imagecW = imageW + padW

            imageSub = imagec[(imagecH-padH):(imagecH+padH+1), (imagecW-padW):(imagecW+padW+1)]
            out[imageH, imageW] = np.sum(imageSub * kernel)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    out = conv_fast(f, np.flip(g[1:,], 1))


    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    meanVal = g[1:].mean()
    updateG = g[1:] - meanVal
    out = conv_fast(f, updateG)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    image = f
    kernel = g
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    padH = 0
    padW = 0

    imagec = image.copy()

    if Hk % 2 != 0:
        padH = int((Hk - 1) / 2)
    else:
        padH = int(Hk / 2)

    if Wk % 2 != 0:
        padW = int((Wk - 1) / 2)
    else:
        padW = int(Wk / 2)

    imagec = zero_pad(imagec, padH, padW)
    kerMean = kernel.mean()
    kerStd = np.std(kerMean)

    for imageH in range(Hi):
        for imageW in range(Wi):
            imagecH = imageH + padH
            imagecW = imageW + padW

            imageSub = imagec[(imagecH-padH):(imagecH+padH+1), (imagecW-padW):(imagecW+padW+1)]
            imageMean = imageSub.mean()
            imageStd = np.std(imageSub)

            out[imageH, imageW] = np.sum((imageSub-imageMean) / imageStd * kernel)
    ### END YOUR CODE

    return out

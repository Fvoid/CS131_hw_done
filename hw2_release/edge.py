import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    for imageH in range(Hi):
        for imageW in range(Wi):
            imagecH = pad_width0 + imageH
            imagecW = pad_width1 + imageW
            imageSub = padded[imagecH - pad_width0 : imagecH + pad_width0 + 1, imagecW - pad_width1 : imagecW + pad_width1 + 1]
            out[imageH, imageW] = np.sum(imageSub * kernel)
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp

    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = (size - 1) // 2
    left = 1 / (2 * np.pi * sigma**2)
    for h in range(size):
        for w in range(size):
            kernel[h, w] = left * np.exp(- ((h-k)**2 + (w-k)**2) / 2 / sigma**2)
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE
    Dx = np.array([[0, 0.0, 0],
                   [0.5, 0.0, -0.5],
                   [0, 0.0, 0]])
    out = conv(img, np.flip(Dx, 1))
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE
    Dy = np.array([[0, 0.5, 0],
                  [0, 0, 0],
                  [0, -0.5, 0]])
    out = conv(img, np.flip(Dy, 0))
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = partial_x(img)
    Gy = partial_y(img)
    matrixSum = np.square(Gx) + np.square(Gy)
    G = np.sqrt(matrixSum)
    theta = np.degrees(np.arctan2(Gy, Gx))

    thetaH, thetaW = theta.shape
    for h in range(thetaH):
        for w in range(thetaW):
            if theta[h, w] < 0:
                theta[h, w] += 360
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    for imageH in range(H):
        for imageW in range(W):
            direction = theta[imageH, imageW]
            inOneH, inOneW = imageH, imageW
            inTwoH, inTwoW = imageH, imageW
            doCal = False
            if direction == 0 or direction == 180 or direction == 360:
                inOneH = imageH
                inOneW = imageW + 1
                inTwoH = imageH
                inTwoW = imageW - 1
                doCal = True
            elif direction == 45 or direction == 225:
                inOneH = imageH - 1
                inOneW = imageW + 1
                inTwoH = imageH + 1
                inTwoW = imageW - 1
                doCal = True
            elif direction == 90 or direction == 270:
                inOneH = imageH - 1
                inOneW = imageW
                inTwoH = imageH + 1
                inTwoW = imageW
                doCal = True
            elif direction == 135 or direction == 315:
                inOneH = imageH - 1
                inOneW = imageW - 1
                inTwoH = imageH + 1
                inTwoW = imageW + 1
                doCal = True

            if doCal:
                if isInRange(inOneH, inOneW, H, W) and isInRange(inTwoH, inTwoW, H, W):
                    if isLargest(G[imageH, imageW], G[inOneH, inOneW], G[inTwoH, inTwoW]):
                        out[imageH, imageW] = G[imageH, imageW]
                elif not isInRange(inOneH, inOneW, H, W) and isInRange(inTwoH, inTwoW, H, W):
                    if isLargest(G[imageH, imageW], 0, G[inTwoH, inTwoW]):
                        out[imageH, imageW] = G[imageH, imageW]
                elif isInRange(inOneH, inOneW, H, W) and not isInRange(inTwoH, inTwoW, H, W):
                    if isLargest(G[imageH, imageW], G[inOneH, inOneW], 0):
                        out[imageH, imageW] = G[imageH, imageW]
                else:
                    out[imageH, imageW] = G[imageH, imageW]

    ### END YOUR CODE

    return out

def isLargest(mid, inOne, inTwo):
    if mid >= inOne and mid >= inTwo:
        return True
    else:
        return False

def isInRange(inH, inW, rangeH, rangeW):
    if 0 <= inH and inH < rangeH and 0 <= inW and inW < rangeW:
        return True
    else:
        return False



def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    #have to specfify dtype to bool
    strong_edges = np.zeros(img.shape, dtype=bool)
    weak_edges = np.zeros(img.shape, dtype = bool)

    ### YOUR CODE HERE

    H, W = img.shape
    for imageH in range(H):
        for imageW in range(W):
            if img[imageH, imageW] >= high:
                strong_edges[imageH, imageW] = True
            elif img[imageH, imageW] >= low and img[imageH, imageW] < high:
                weak_edges[imageH, imageW] = True
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))


    ### YOUR CODE HERE
    for h, w in indices:
        edges[h, w] = 1
        neighbor = get_neighbors(h, w, H, W)
        for neighborH, neighborW in neighbor:
            if weak_edges[neighborH, neighborW]:
                edges[neighborH, neighborW] = 1
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    print('. ', end="")
    ### YOUR CODE HERE
    #Step 1 smoothing
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)

    #step 2 find gradient
    G, theta = gradient(smoothed)

    #step 3 non-maximum suppression
    nms = non_maximum_suppression(G, theta)

    #step 4 double_thresholding
    strong_edges, weak_edges = double_thresholding(nms, high, low)

    #edge tracking
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)

    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        for t in range(num_thetas):
            rho = int(round(x * cos_t[t] + y * sin_t[t])) + diag_len
            accumulator[rho, t] += 1
    ### END YOUR CODE

    return accumulator, rhos, thetas

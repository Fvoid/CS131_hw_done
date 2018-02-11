import numpy as np
from skimage import filters
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve, 
        which is already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    dx_sq = np.square(dx)
    dy_sq = np.square(dy)
    dxy = dx * dy
    Sx2 = convolve(dx_sq, window)
    Sy2 = convolve(dy_sq, window)
    Sxy = convolve(dxy, window)
    response = Sx2 * Sy2 - np.square(Sxy) - k * np.square(Sx2 + Sy2)
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = []
    ### YOUR CODE HERE
    pMax = patch.max()
    pMin = patch.min()
    
    aNorm = np.linalg.norm(patch)
    tmp = patch / aNorm
    feature = tmp.flatten()
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []
    
    M = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    print(desc1.shape)
    print(desc2.shape)
    print(dists.shape)
    for i in range(M):
        aRow = dists[i,:]
        smallest = np.partition(aRow, 0)[0]
        secondSm = np.partition(aRow, 1)[1]
        if smallest / secondSm <= threshold:
           matches.append([i, np.argwhere(aRow == smallest)[0,0]])
            
    ### END YOUR CODE
    
    return np.array(matches)


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1 
    
    Hint:
        You can use np.linalg.lstsq function to solve the problem. 
        
    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)
        
    Return:
        H: a matrix of shape (P * P) that transform p2 to p1.
    https://stackoverflow.com/questions/20546182/how-to-perform-coordinates-affine-transformation-using-python-part-2/20555267
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    ### YOUR CODE HERE
    
    H, res, rank, x = np.linalg.lstsq(p2, p1) 
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N, dtype="bool")
    n_inliers = 0
    
    # RANSAC iteration start
    ### YOUR CODE HERE
    iteration = 0
    while iteration < n_iters:
        #matched2 is input, matched1 is output
        #partition
        all_index = np.arange(N)
        train_index, test_index = random_partition(n_samples, all_index)
        train_data_in = matched2[train_index]
        train_data_out = matched1[train_index]
        test_data_in = matched2[test_index]
        test_data_out = matched1[test_index]
        
        #compute affine transformation
        maybeModel = fit_in_model(train_data_in, train_data_out)
        
        #compute inliers
        test_error = compute_error(test_data_in, test_data_out, maybeModel)
        also_index = test_index[test_error < threshold]
        cur_inliers = len(also_index) + n_samples
        if cur_inliers > n_inliers:
            n_inliers = cur_inliers
            max_inliers[also_index] = True
            max_inliers[train_index] = True
            input_inliers = np.concatenate((train_data_in, matched2[also_index]))
            output_inliers = np.concatenate((train_data_out, matched1[also_index]))
            H = fit_in_model(input_inliers, output_inliers)
        
        iteration += 1
        
    ### END YOUR CODE
    return H, matches[max_inliers]

def compute_error(inputData, expectData, model):
    actual = np.dot(inputData, model)
    error = np.sum((expectData - actual)**2, axis=1)
    return error

def fit_in_model(inputVal, expectOutVal):
    H, res, rank, x = np.linalg.lstsq(inputVal, expectOutVal)
    H[:,2] = np.array([0, 0, 1])
    return H

def random_partition(n, data):
    """
    partition data into traning set and test set
    from 0 to n row is training set, the rest is test set
    """
    np.random.shuffle(data)
    training = data[:n]
    testing = data[n:]
    return training, testing

def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. compute the gradient image in x and y (already done for you)
    2. compute gradient histograms
    3. normalize across block 
    4. flattening block into a feature vector

    Args:
        patch: grayscale image patch of shape (h, w)
        pixels_per_cell: size of a cell with shape (m, n)

    Returns:
        block: 1D array of shape ((h*w*n_bins)/(m*n))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)
   
    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    #I copy someone's solution ~
    block = np.zeros(shape=(rows, cols, n_bins), dtype=np.float32)
    for aRow in range(rows):
        for aCol in range(cols):
            G_cells[aRow, aCol] = G_cells[aRow, aCol] / np.linalg.norm(G_cells[aRow, aCol])
            for h in range(pixels_per_cell[0]):
                for w in range(pixels_per_cell[1]):
                    b0 = theta_cells[aRow, aCol][h,w] // degrees_per_bin
                    if b0 >= n_bins:
                        b0 = 0
                    b1 = b0 + 1
                    if b1 >= n_bins:
                        b1 = 0
                    b = theta_cells[aRow, aCol][h, w] % degrees_per_bin / degrees_per_bin
                    block[aRow, aCol][int(b0)] += 1 - b
                    block[aRow, aCol][int(b1)] += b
    block = block / np.sum(block)
    block = block.reshape(rows*cols*n_bins)
                    
    ### YOUR CODE HERE
    
    return block

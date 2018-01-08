import numpy as np
import cv2

def threshold(img, s_thresh=(100, 255), sx_thresh=(20, 255), l_thresh=(112, 255), use_sharr=True, rgb=False):
    """Apply threshold to image based on color and/or gradient.
    Inspired to https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/a1b70df9-638b-46bb-8af0-12c43dcfd0b4
    """

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    if use_sharr:
        sobelx = cv2.Scharr(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    else:
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * (abs_sobelx-abs_sobelx.min()) / (abs_sobelx.max()-abs_sobelx.min()))

    # Threshold x gradient
    sxbinary = np.uint8((scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]))

    # Threshold color channel
    s_binary = np.uint8((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]))

    # Threshold luminance channel
    l_binary = np.uint8((l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1]))

    if rgb:
        return np.stack((s_binary, l_binary, sxbinary), axis=2) * np.uint8(255)
    else:
        return sxbinary | (s_binary & l_binary)

def get_rect_from_centroid(centroid, width, height):
    return (centroid[0] - width//2, centroid[1] - height//2), width, height

def find_window_centroids(image, window_width=50, window_height=80, margin=100, mass_center=True, polyfit=None, xoffset=0):
    window_centroids = [] # Store the window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the starting position either by using the provided polynomial
    # or by using np.sum to get the vertical image slice and then np.convolve
    # the vertical image slice with the window template

    if polyfit is not None:
        center = np.polyval(polyfit, image.shape[0]) - xoffset
        current_margin = margin
    else:
        # Sum quarter bottom of image to get slice, could use a different ratio
        y_start = int(3*image.shape[0]/4)
        y_end = image.shape[0]
        sum = np.sum(image[y_start:y_end, :], axis=0)
        if np.all(sum == 0):
            center = image.shape[1]//2
            current_margin = center
        else:
            conv_signal = np.convolve(window, sum)
            center = np.argmax(conv_signal) - window_width/2
            current_margin = margin

    # Go through each layer looking for max pixel locations
    for level in range(0, int(image.shape[0]/window_height)):
        if polyfit is not None:
            center = np.polyval(polyfit, image.shape[0] - (level+0.5)*window_height) - xoffset
            current_margin = margin
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0] - (level+1)*window_height):
                                   int(image.shape[0] - level*window_height), :],
                             axis=0)
        conv_signal = np.convolve(window, image_layer, mode='same')

        # Find the best centroid by using past center as a reference
        min_index = int(max(center - current_margin, 0))
        max_index = int(min(center + current_margin, image.shape[1]))

        if conv_signal[min_index:max_index].sum() == 0:
            # No edges in this layer, skip to next but increase margin
            current_margin += margin
            continue
        # Edges found, find the maximum and reset the margin
        center = np.argmax(conv_signal[min_index:max_index]) + min_index
        current_margin = margin

        if mass_center:
            # Move to mass center of the window
            min_index = int(max(0, center-window_width))
            max_index = int(min(center+window_width, image.shape[1]))
            x = np.arange(min_index, max_index)
            center = np.average(x, weights=image_layer[min_index:max_index])

        # Add what we found for that layer
        window_centroids.append((center,
                                 int(image.shape[0] - (level+0.5)*window_height),
                                 conv_signal[int(center)],
        ))

    return np.transpose(window_centroids) if window_centroids else (None,)*3

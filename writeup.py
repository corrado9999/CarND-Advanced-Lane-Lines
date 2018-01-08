import glob
import numpy as np
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.patches as mpl_patches
import pygments
import inspect
from IPython.display import HTML
from moviepy.editor import VideoFileClip

import camera
import edges
import lane as lane_m
import utils

plt.rcParams['figure.dpi'] = 50
plt.rcParams['figure.figsize'] = (18,9)

python_lexer = pygments.lexers.PythonLexer()
html_formatter = pygments.formatters.HtmlFormatter(full=True)

calibrate = camera.Calibration(glob.glob('./camera_cal/*'))

corners1 = np.array([(205,720), (574.5,465), (709,465), (1103,720)]) #estimated on straight_lines1.jpg
corners2 = np.array([(220,720), (573,  465), (713,465), (1110,720)]) #estimated on straight_lines2.jpg
src_corners = (corners1 + corners2)/2. #average the two estimations
dest_corners = np.array([(     280, 720), (     280,   0),
                         (1280-280,   0), (1280-280, 720)], dtype=float)

straighten = camera.PerspectiveTransformation(src_corners, dest_corners)
dash = np.array([[747, 777], [488,  507]])
center = np.array([src_corners[::3, 0].mean(), 668])
#src_corners[::3, :] = np.round(np.transpose(straighten.inverse(dest_corners[::3, 0],
#                                                               straighten(*center)[1])))
#straighten = camera.PerspectiveTransformation(src_corners, dest_corners)

xm_per_pix = 3.7 / (dest_corners[-1,0] - dest_corners[0,0]) # meters per pixel in x dimension
ym_per_pix = 3.0 / np.diff(straighten(*dash))[1,0]          # meters per pixel in y dimension

lane = lane_m.Lane(straighten, xm_per_pix=xm_per_pix, ym_per_pix=ym_per_pix)

def camera_calibration(test_image):
    img = cv2.imread(test_image)[..., ::-1]
    plt.subplot(121); plt.imshow(img)
    plt.subplot(122); plt.imshow(calibrate(img))
    plt.tight_layout()


def image_thresholding(test_image, rgb):
    img = calibrate(cv2.imread(test_image)[..., ::-1])

    plt.subplot(121)
    plt.imshow(img)
    plt.title(test_image)

    plt.subplot(122)
    mask = edges.threshold(img, rgb=rgb)
    plt.imshow(mask, cmap='gray')

    plt.tight_layout()

def perspective(test_image, with_dash=False):
    img = calibrate(cv2.imread(test_image)[...,::-1])

    plt.subplot(121)
    plt.imshow(img)
    plt.plot(*np.transpose(straighten.source_corners.tolist() + [straighten.source_corners[0]]), 'r')
    if with_dash:
        plt.plot(*dash, 'bo', markerfacecolor='none', markersize=15)
    plt.ylim(img.shape[0], 0)
    plt.title(test_image)

    plt.subplot(122)
    img = straighten(img)
    plt.imshow(img)
    plt.plot(*straighten.dest_corners.T, 'r')
    if with_dash:
        plt.plot(*straighten(*dash), 'bo', markerfacecolor='none', markersize=15)
    plt.ylim(img.shape[0], 0)

    plt.tight_layout()
    plt.show()

def lines_detection(test_image, topdown_view=True, with_info=False):
    rgb = cv2.imread(test_image)[..., ::-1]
    calibrated = calibrate(rgb)
    mask = lane.preprocess_image(calibrated)

    lane.reset()
    lane.update(calibrated)

    if topdown_view:
        image = lane.draw_lane(straighten(calibrated), topdown_view=True)
    else:
        image = lane.draw_lane(calibrated, topdown_view=False)
    if with_info:
        image = lane.draw_info(image)

    mask = np.stack([mask]*3, axis=2)
    mask = lane.plot_centroids(mask, topdown_view=True)
    mask = lane.draw_lines(mask, topdown_view=True)

    plt.subplot(121)
    plt.imshow(image)
    plt.title(test_image)

    plt.subplot(122)
    plt.imshow(mask, cmap='gray')

    plt.tight_layout()
    plt.show()

def get_source(*objs):
    code = '\n'.join(inspect.getsource(obj) for obj in objs)
    return HTML(pygments.highlight(code, python_lexer, html_formatter))

def process_image(image):
    image = calibrate(image)
    lane.update(image)
    image = lane.draw_lane(image)
    image = lane.draw_info(image)
    return image

def process_video(videofile, subclip=None, alpha=None):
    lane.reset()
    alpha_orig = lane.left_line.alpha, lane.rght_line.alpha
    if alpha is not None:
        lane.alpha = alpha
    try:
        output = 'test_videos_output/' + videofile
        output = output.rsplit('.')[0] + '.avi'
        input_clip = VideoFileClip(videofile)
        if subclip:
            input_clip = input_clip.subclip(*subclip)
        output_clip = input_clip.fl_image(process_image)
        output_clip.write_videofile(output, audio=False, codec='png')
    except:
        lane.left_line.alpha, lane.rght_line.alpha = alpha_orig
        raise

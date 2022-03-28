import cv2
import numpy as np
from baseImage import Image, Rect
from image_registration.matching.keypoint.Kaze import Kaze
import time


a = Kaze()

im_source = Image('tests/image/0.png')
im_search = Image('tests/image/0.png').crop(Rect(100, 100, 200, 200))
a.find_best_result(im_source, im_search)

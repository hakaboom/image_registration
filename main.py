import cv2
import numpy as np
from baseImage import Image, Rect
from image_registration.matching.keypoint.Kaze import Kaze
from image_registration.matching.keypoint.orb import ORB
from image_registration.matching.keypoint.sift import SIFT
from image_registration.utils import keypoint_angle, get_keypoint_from_matches, keypoint_origin_angle
import time
import math
import cupy as cp

a = ORB(rgb=False)
# im_source = Image('tests/image/4.png', place=1)#.rotate(cv2.ROTATE_180)
# im_search = Image('tests/image/3.png', place=1).crop(Rect(1827, 69, 52, 53))


im_source = Image('tests/image/9.png', place=3)#.cvtColor(cv2.COLOR_BGR2GRAY)
im_search = Image('tests/image/2.png', place=3).crop(Rect(1257, 532, 135, 112))#.cvtColor(cv2.COLOR_BGR2GRAY)

for i in range(1000):
    start = time.time()
    a.find_best_result(im_source, im_search, rgb=False)
    print(time.time() - start)

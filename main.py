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


a = SIFT()
# im_source = Image('tests/image/4.png', place=1)#.rotate(cv2.ROTATE_180)
# im_search = Image('tests/image/3.png', place=1).crop(Rect(1827, 69, 52, 53))

im_source = Image('tests/image/8.png', place=1)#.rotate(cv2.ROTATE_180) 2 3 4? 7 10? 11 12? 13? 15?
im_search = Image('tests/image/7.png', place=1).crop(Rect(1166, 440, 120, 268))
for i in range(1):
    start = time.time()
    print(a.find_best_result(im_source, im_search))
    # print(time.time() - start)
#

# query_point_angle = 30   2 3 5
# query_point_quadrant, query_point_angle = divmod(query_point_angle, 90)
# query_point_quadrant = (4, 3, 2, 1)[query_point_quadrant]

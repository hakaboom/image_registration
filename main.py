import cv2
import numpy as np
from baseImage import Image, Rect
from image_registration.matching.keypoint.Kaze import Kaze
from image_registration.matching.keypoint.orb import ORB
from image_registration.matching.keypoint.sift import SIFT
from image_registration.matching.keypoint.surf import SURF
from image_registration.utils import keypoint_angle, get_keypoint_from_matches, keypoint_origin_angle
import time
import math


a = SIFT()
# im_source = Image('tests/image/3.png', place=1)#.rotate(cv2.ROTATE_180)
# im_search = Image('tests/image/3.png', place=1).crop(Rect(1827, 69, 52, 53))

#
im_source = Image('tests/image/11.png', place=1)
im_search = Image('tests/image/11.png', place=1).crop(Rect(351,283,151,143))
# # im_source.crop(Rect(549, 366, 145, 269)).imwrite('source.png')
# im_search.imshow('search')
# # cv2.waitKey(0)
for i in range(1):
    start = time.time()
    print(a.find_all_result(im_source, im_search))
    # print(time.time() - start)

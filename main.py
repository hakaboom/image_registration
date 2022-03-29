import cv2
import numpy as np
from baseImage import Image, Rect
from image_registration.matching.keypoint.Kaze import Kaze
from image_registration.matching.keypoint.orb import ORB
from image_registration.matching.keypoint.sift import SIFT
import time


a = SIFT()
#
im_source = Image('tests/image/2.png', place=1)
# im_source.rectangle(rect=Rect(1244, 108, 31, 30), color=(0, 0, 0), thickness=-1)
im_search = Image('tests/image/2.png', place=1)# .crop(Rect(1244, 108, 32, 31))

for i in range(1):
    start = time.time()
    print(a.find_best_result(im_source, im_search, rgb=True))
    # print(time.time() - start)
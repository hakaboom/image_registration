import cv2
import numpy as np
from baseImage import Image, Rect
from image_registration.matching import SIFT, ORB, SURF, AKAZE, MatchTemplate
from image_registration.utils import keypoint_distance, rectangle_transform
import time
import math
import os
import re

from pathlib import Path
import pathlib

match = SIFT(rgb=False)
im_source = Image('tests/image/6.png', place=1).rotate(cv2.ROTATE_180)
im_search = Image('tests/image/4.png', place=1).crop(Rect(1498,68,50,56))
# im_search = Image('tests/image/6.png', place=1).crop(Rect(500, 100, 200, 200))

#
# im_source = Image('tests/image/8.png', place=0)
# im_search = Image('tests/image/7.png', place=0).crop(Rect(1393,54,22,20))
#
#
for i in range(1):
    start = time.time()
    ret = match.find_all_results(im_source, im_search)
    print(ret)
    test = im_source.clone()
    for _ in ret:
        test.rectangle(rect=_['rect'], color=(0, 0, 255), thickness=3)
    test.imshow('ret')
    cv2.waitKey(0)


# r'C:\Users\Administrator.hzq\Desktop\test\Main\1265.png'

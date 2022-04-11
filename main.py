import cv2
import numpy as np
from baseImage import Image, Rect
from image_registration.matching import SIFT, ORB, SURF, AKAZE
import time
import math

#
a = AKAZE()
im_source = Image('tests/image/3.png', place=1)#.rotate(cv2.ROTATE_180)
im_search = Image('tests/image/3.png', place=1).crop(Rect(0, 0, 200, 200))

#
# im_source = Image('tests/image/8.png', place=0)
# im_search = Image('tests/image/7.png', place=0).crop(Rect(1393,54,22,20))
#
#
for i in range(1):
    start = time.time()
    ret = a.find_all_result(im_source, im_search)

    test = im_source.clone()
    for _ in ret:
        test.rectangle(rect=_['rect'], color=(255, 255, 0), thickness=3)
    test.imshow('ret')
    cv2.waitKey(0)

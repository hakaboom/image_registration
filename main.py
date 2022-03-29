import cv2
import numpy as np
from baseImage import Image, Rect
from image_registration.matching.keypoint.Kaze import Kaze
from image_registration.matching.keypoint.orb import ORB
from image_registration.matching.keypoint.sift import SIFT
import time


a = ORB()


im_source = Image('tests/image/2.png', place=3)
# im_source.rectangle(rect=Rect(1244, 108, 31, 30), color=(0, 0, 0), thickness=-1)
im_search = Image('tests/image/2.png', place=3)# .crop(Rect(1244, 108, 32, 31))

for i in range(100):
    start = time.time()
    a.get_keypoint_and_descriptor(im_source)
    print(time.time() - start)
import cv2
import numpy as np
from baseImage import Image, Rect
from image_registration.matching.template.matchTemplate import MatchTemplate
import time
#
# match = MatchTemplate(rgb=True)

im1 = Image('tests/image/1.png', place=1).cvtColor(cv2.COLOR_BGR2GRAY)
im2 = Image('tests/image/1.png', place=1).crop(Rect(1244, 108, 32, 31)).cvtColor(cv2.COLOR_BGR2GRAY)
matcher = cv2.matchTemplate

result = matcher(im1.data, im2.data, cv2.TM_CCOEFF_NORMED)

print(cv2.minMaxLoc(result))
im1_2 = Image('tests/image/1.png', place=3).cvtColor(cv2.COLOR_BGR2GRAY)
im2_2 = Image('tests/image/1.png', place=3).crop(Rect(1244, 108, 32, 31)).cvtColor(cv2.COLOR_BGR2GRAY)

result2 = matcher(im1_2.data, im2_2.data, cv2.TM_CCOEFF_NORMED).get()
cv2.minMaxLoc(result2)
print(cv2.minMaxLoc(result2))
# im_search = Image('tests/image/1.png', place=1).crop(Rect(1244, 108, 32, 31))
#
# res2 = match.find_best_result(im_source, im_search)

import cv2
import time
from baseImage import Image, Rect
from baseImage.utils.ssim import SSIM
from image_registration.matching import SIFT, ORB, CudaMatchTemplate, CUDA_ORB

match = CUDA_ORB()
im_source = Image('tests/image/6.png', place=2)
im_search = Image('tests/image/6.png', place=2).crop(Rect(0, 0, 300, 300))

start = time.time()


result = match.find_all_results(im_source, im_search)

# img = im_source.clone()
# for _ in result:
#     img.rectangle(rect=_['rect'], color=(0, 0, 255), thickness=3)
# img.imshow('ret')
# cv2.waitKey(0)
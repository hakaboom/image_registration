#! usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
from typing import Union, Tuple, List

from baseImage import Image

from image_registration.matching.keypoint.base import BaseKeypoint


image_type = Union[str, bytes, np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat, Image]
keypoint_type = Tuple[cv2.KeyPoint, ...]
matches_type = Tuple[Tuple[cv2.DMatch, ...], ...]
good_match_type = List[cv2.DMatch]


class SURF(BaseKeypoint):
    FLANN_INDEX_KDTREE: int = 0

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True,
                 hessianThreshold: int = 400, nOctaves: int = 4, nOctaveLayers: int = 3,
                 extended: bool = True, upright: bool = False): ...

    def create_detector(self, **kwargs) -> cv2.xfeatures2d.SURF: ...
#! usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
from baseImage.constant import Place

from image_registration.matching.keypoint.base import BaseKeypoint
from typing import Union


class ORB(BaseKeypoint):
    METHOD_NAME = 'ORB'
    Dtype = np.uint8
    Place = (Place.Mat, Place.UMat, Place.Ndarray)

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True, **kwargs):
        super(ORB, self).__init__(threshold=threshold, rgb=rgb, **kwargs)

    def create_matcher(self) -> cv2.DescriptorMatcher:
        """
        创建特征点匹配器

        Returns:
            cv2.FlannBasedMatcher
        """
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        return matcher

    def create_detector(self) -> cv2.ORB:
        detector = cv2.ORB_create()
        return detector

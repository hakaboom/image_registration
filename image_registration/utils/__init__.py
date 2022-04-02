#! usr/bin/python
# -*- coding:utf-8 -*-
import math
import cv2


def generate_result(rect, confi):
    """Format the result: 定义图像识别结果格式."""
    ret = {
        'rect': rect,
        'confidence': confi,
    }
    return ret


def keypoint_distance(kp1: cv2.KeyPoint, kp2: cv2.KeyPoint):
    """求两个keypoint的两点之间距离"""
    x = kp1.pt[0] - kp2.pt[0]
    y = kp1.pt[1] - kp2.pt[1]
    return math.sqrt((x ** 2) + (y ** 2))


def keypoint_angle(kp1: cv2.KeyPoint, kp2: cv2.KeyPoint):
    """求两个keypoint的夹角 """
    k = [
        (kp1.angle - 180) if kp1.angle >= 180 else kp1.angle,
        (kp2.angle - 180) if kp2.angle >= 180 else kp2.angle
    ]
    if k[0] == k[1]:
        return 0
    else:
        return abs(k[0] - k[1])

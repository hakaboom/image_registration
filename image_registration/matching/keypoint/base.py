#! usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import cupy as cp
import time
import math
from baseImage import Image, Rect, Point
from baseImage.constant import Place

from image_registration.matching import MatchTemplate
from image_registration.utils import generate_result, keypoint_distance, keypoint_angle, get_keypoint_from_matches, keypoint_origin_angle
from image_registration.exceptions import NoEnoughPointsError, PerspectiveTransformError, HomographyError, MatchResultError
from typing import Union, Tuple, List


class BaseKeypoint(object):
    FILTER_RATIO = 1
    METHOD_NAME = None
    Dtype = None
    Place = None
    template = MatchTemplate()

    def __init__(self, threshold=0.8, rgb=True, **kwargs):
        """
        初始化

        Args:
            threshold: 识别阈值(0~1)
            rgb: 是否使用rgb通道进行校验
        """
        self.threshold = threshold
        self.rgb = rgb
        self.detector = self.create_detector(**kwargs)
        self.matcher = self.create_matcher(**kwargs)

    def create_matcher(self, **kwargs):
        raise NotImplementedError

    def create_detector(self, **kwargs):
        raise NotImplementedError

    def find_best_result(self, im_source, im_search, threshold=None, rgb=None):
        """
        通过特征点匹配,在im_source中找到最符合im_search的范围

        Args:
            im_source: 待匹配图像
            im_search: 图片模板
            threshold: 识别阈值(0~1)
            rgb: 是否使用rgb通道进行校验

        Returns:

        """
        threshold = threshold or self.threshold
        rgb = rgb or self.rgb

        im_source, im_search = self.input_image_check(im_source, im_search)
        if im_source.channels == 1:
            rgb = False

        kp_src, des_src = self.get_keypoint_and_descriptor(image=im_source)
        kp_sch, des_sch = self.get_keypoint_and_descriptor(image=im_search)
        # 在特征点集中,匹配最接近的特征点
        rect, matches, good = self.get_rect_from_good_matches(im_source=im_source, im_search=im_search, kp_src=kp_src, des_src=des_src,
                                                              kp_sch=kp_sch, des_sch=des_sch)
        if not rect:
            return None
        # 根据识别的结果,从待匹配图像中截取范围,进行模板匹配求出相似度
        confidence = self._cal_confidence(im_source=im_source, im_search=im_search, crop_rect=rect, rgb=rgb)
        best_match = generate_result(rect=rect, confi=confidence)
        return best_match if confidence > threshold else None

    def get_keypoint_and_descriptor(self, image: Image):
        """
        获取图像关键点(keypoint)与描述符(descriptor)

        Args:
            image: 待检测的灰度图像

        Returns:

        """
        if image.channels == 3:
            image = image.cvtColor(cv2.COLOR_BGR2GRAY).data
        else:
            image = image.data
        keypoint, descriptor = self.detector.detectAndCompute(image, None)

        if len(keypoint) < 2:
            raise NoEnoughPointsError('{} detect not enough feature points in input images'.format(self.METHOD_NAME))
        return keypoint, descriptor

    def get_rect_from_good_matches(self, im_source, im_search, kp_src, des_src, kp_sch, des_sch):
        """
        从特征点里获取最佳的范围

        Returns:
            rect, matches, good
        """
        start = time.time()
        matches = self.match_keypoint(des_sch=des_sch, des_src=des_src)
        good = self.get_good_in_matches(matches=matches)
        # 按照queryIdx排升序
        good = sorted(good, key=lambda x: x.queryIdx)
        # print(f'good={len(good)}, kp_src={len(kp_src)}, kp_sch={len(kp_sch)}')
        # _test = []
        # r = Rect(1041, 667, 139, 116)
        # for i in good:
        #     point = kp_src[i.trainIdx]
        #     if r.contains(Point(point.pt[0], point.pt[1])):
        #         _test.append(i)
        # 筛选重复的queryidx
        queryidx_list = []
        # queryidx_list的索引对应的queryidx
        queryidx_index_list = []
        queryidx_index = 0
        queryidx_flag = True
        # 假设第一个点,及distance最小的点,为基准点
        first_good_point = good[0]  # 随便填一个用于对比
        while queryidx_flag:
            point = good[queryidx_index]
            _queryIdx = point.queryIdx
            queryidx_index_list.append(_queryIdx)
            first_good_point = first_good_point if point.distance > first_good_point.distance else point
            point_list = [point]
            while True:
                queryidx_index += 1
                if queryidx_index == len(good):
                    queryidx_flag = False
                    break
                new_point = good[queryidx_index]
                new_queryidx = new_point.queryIdx
                if _queryIdx == new_queryidx:
                    point_list.append(new_point)
                else:
                    break
            queryidx_list.append(np.array(point_list))

        first_good_point_train: cv2.KeyPoint = kp_src[first_good_point.trainIdx]
        first_good_point_query: cv2.KeyPoint = kp_sch[first_good_point.queryIdx]
        first_good_point_query_index = queryidx_index_list.index(first_good_point.queryIdx)
        first_good_point_angle = first_good_point_train.angle - first_good_point_query.angle

        Image(cv2.drawMatches(im_search.data, kp_sch, im_source.data, kp_src, (first_good_point,), None, flags=2)).\
            imshow('first_good')

        # 计算模板图像上,该点与其他特征点的夹角
        # first_good_point_sch_angle = [keypoint_angle(kp1=first_good_point_query, kp2=i) for i in kp_sch]
        first_good_point_sch_origin_angle = []
        for i in kp_sch:
            _angle = keypoint_origin_angle(kp1=first_good_point_query, kp2=i)
            if _angle != 0:
                _angle = _angle - first_good_point_query.angle
            first_good_point_sch_origin_angle.append(_angle)

        # 计算目标图像中,该点与其他特征点的夹角
        good_point = []
        for i in queryidx_list:
            query_point = kp_sch[i[0].queryIdx]
            # 根据first_good_point的旋转,计算其他特征点旋转后的角度
            query_rotate_angle = query_point.angle + first_good_point_angle
            train_points = get_keypoint_from_matches(kp_src, i, 'train')
            train_points_angle = np.array([i.angle for i in train_points])
            if query_rotate_angle >= 360:
                query_rotate_angle -= 360
            angle_gap = np.abs(train_points_angle - query_rotate_angle)
            sort_angle_gap = np.argsort(angle_gap)
            test_angle = [keypoint_angle(kp1=first_good_point_query, kp2=i) for i in train_points]
            good_point.append(i[sort_angle_gap[0]])

        # 计算各点以first_good_point为原点的旋转角
        good_point_keypoint = get_keypoint_from_matches(kp_src, good_point, 'train')
        ret_keypoint = []
        ret_keypoint_pt = []
        ret = []
        origin_angle_threshold = round(5 / 360, 2) * 100  # 允许的偏差值,x表示角度 round(x / 360, 2) * 100
        for i, keypoint in enumerate(good_point_keypoint):
            _angle = keypoint_origin_angle(kp1=first_good_point_train, kp2=keypoint)
            if _angle != 0:
                _angle = _angle - first_good_point_train.angle
            sch_origin_angle = first_good_point_sch_origin_angle[queryidx_index_list[i]]
            if round(abs(_angle - sch_origin_angle) / 360, 2) * 100 < origin_angle_threshold:
                if keypoint.pt not in ret_keypoint_pt:  # 去重
                    ret_keypoint.append(keypoint)
                    ret_keypoint_pt.append(keypoint.pt)
                    ret.append(good_point[i])

        Image(cv2.drawMatches(im_search.data, kp_sch, im_source.data, kp_src, ret, None, flags=2)).imshow('ret')
        Image(cv2.drawMatches(im_search.data, kp_sch, im_source.data, kp_src, good, None, flags=2)).imshow('good')
        # cv2.waitKey(0)
        rect = self.extract_good_points(im_source=im_source, im_search=im_search, kp_sch=kp_sch, kp_src=kp_src, good=ret)
        return rect, matches, good

    def match_keypoint(self, des_sch, des_src, k=20):
        """
        特征点匹配

        Args:
            des_src: 待匹配图像的描述符集
            des_sch: 图片模板的描述符集
            k(int): 获取多少匹配点

        Returns:
            List[List[cv2.DMatch]]: 包含最匹配的描述符
        """
        # k=2表示每个特征点取出2个最匹配的对应点
        matches = self.matcher.knnMatch(des_sch, des_src, k)
        return matches

    def get_good_in_matches(self, matches):
        """
        特征点过滤

        Args:
            matches: 特征点集

        Returns:
            List[cv2.DMatch]: 过滤后的描述符集
        """
        if not matches:
            return None
        good = []
        for match_index in range(len(matches)):
            match = matches[match_index]
            for DMatch_index in range(len(match)):
                if match[DMatch_index].distance <= self.FILTER_RATIO * match[-1].distance:
                    good.append(match[DMatch_index])
        return good

    def extract_good_points(self, im_source, im_search, kp_src, kp_sch, good):
        """
        根据匹配点(good)数量,提取识别区域

        Args:
            im_source: 待匹配图像
            im_search: 图片模板
            kp_src: 关键点集
            kp_sch: 关键点集
            good: 描述符集

        Returns:

        """
        len_good = len(good)
        if len_good in [0, 1]:
            return None
        elif len_good in [2, 3]:
            # TODO: 待做
            pass
        else:
            return self._many_good_pts(im_source=im_source, im_search=im_search,
                                       kp_sch=kp_sch, kp_src=kp_src, good=good)

    def _many_good_pts(self, im_source: Image, im_search: Image, kp_sch: List[cv2.KeyPoint], kp_src: List[cv2.KeyPoint],
                       good: List[cv2.DMatch]) -> Rect:
        """
        特征点匹配数量>=4时,使用单矩阵映射,求出识别的目标区域

        Args:
            im_source: 待匹配图像
            im_search: 图片模板
            kp_sch: 关键点集
            kp_src: 关键点集
            good: 描述符集

        Returns:

        """
        sch_pts, img_pts = np.float32([kp_sch[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2), np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # M是转化矩阵
        M, mask = self._find_homography(sch_pts, img_pts)
        # 计算四个角矩阵变换后的坐标，也就是在大图中的目标区域的顶点坐标:
        h, w = im_search.shape[:2]
        h_s, w_s = im_source.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        try:
            dst = cv2.perspectiveTransform(pts, M)
        except cv2.error as err:
            raise PerspectiveTransformError(err)
        img = im_source.clone().data
        img2 = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        Image(img).imshow()
        # cv2.waitKey(0)
        def cal_rect_pts(_dst):
            return [tuple(npt[0]) for npt in np.rint(_dst).astype(np.float).tolist()]

        pypts = cal_rect_pts(dst)
        # pypts四个值按照顺序分别是: 左上,左下,右下,右上
        # 注意：虽然4个角点有可能越出source图边界，但是(根据精确化映射单映射矩阵M线性机制)中点不会越出边界
        lt, br = pypts[0], pypts[2]
        # 考虑到算出的目标矩阵有可能是翻转的情况，必须进行一次处理，确保映射后的“左上角”在图片中也是左上角点：
        x_min, x_max = min(lt[0], br[0]), max(lt[0], br[0])
        y_min, y_max = min(lt[1], br[1]), max(lt[1], br[1])
        # 挑选出目标矩形区域可能会有越界情况，越界时直接将其置为边界：
        # 超出左边界取0，超出右边界取w_s-1，超出下边界取0，超出上边界取h_s-1
        # 当x_min小于0时，取0。  x_max小于0时，取0。
        x_min, x_max = int(max(x_min, 0)), int(max(x_max, 0))
        # 当x_min大于w_s时，取值w_s-1。  x_max大于w_s-1时，取w_s-1。
        x_min, x_max = int(min(x_min, w_s - 1)), int(min(x_max, w_s - 1))
        # 当y_min小于0时，取0。  y_max小于0时，取0。
        y_min, y_max = int(max(y_min, 0)), int(max(y_max, 0))
        # 当y_min大于h_s时，取值h_s-1。  y_max大于h_s-1时，取h_s-1。
        y_min, y_max = int(min(y_min, h_s - 1)), int(min(y_max, h_s - 1))
        return Rect(x=x_min, y=y_min, width=(x_max - x_min), height=(y_max - y_min))

    def _cal_confidence(self, im_source, im_search, crop_rect: Rect, rgb: bool) -> Union[int, float]:
        """
        将截图和识别结果缩放到大小一致,并计算可信度

        Args:
            im_source: 待匹配图像
            im_search: 图片模板
            crop_rect: 需要在im_source截取的区域
            rgb:是否使用rgb通道进行校验

        Returns:

        """
        try:
            target_img = im_source.crop(crop_rect)
        except OverflowError:
            raise MatchResultError(f"Target area({crop_rect}) out of screen{im_source.size}")

        h, w = im_search.size
        target_img.resize(w, h)
        if rgb:
            confidence = self.template.cal_rgb_confidence(im_source=im_search, im_search=target_img)
        else:
            confidence = self.template.cal_ccoeff_confidence(im_source=im_search, im_search=target_img)

        confidence = (1 + confidence) / 2
        return confidence

    def input_image_check(self, im_source, im_search):
        im_source = self._image_check(im_source)
        im_search = self._image_check(im_search)

        assert im_source.place == im_search.place, '输入图片类型必须相同, source={}, search={}'.format(im_source.place, im_search.place)
        assert im_source.dtype == im_search.dtype, '输入图片数据类型必须相同, source={}, search={}'.format(im_source.dtype, im_search.dtype)
        assert im_source.channels == im_search.channels, '输入图片通道必须相同, source={}, search={}'.format(im_source.channels, im_search.channels)

        return im_source, im_search

    def _image_check(self, data):
        if not isinstance(data, Image):
            data = Image(data, dtype=self.Dtype)

        if data.place not in self.Place:
            raise TypeError('Image类型必须为(Place.Mat, Place.UMat, Place.Ndarray)')
        return data

    @staticmethod
    def _find_homography(sch_pts, src_pts):
        """
        多组特征点对时，求取单向性矩阵
        """
        try:
            M, mask = cv2.findHomography(sch_pts, src_pts, cv2.RANSAC, 5.0)
        except cv2.error:
            import traceback
            traceback.print_exc()
            raise HomographyError("OpenCV error in _find_homography()...")
        else:
            if mask is None:
                raise HomographyError("In _find_homography(), find no mask...")
            else:
                return M, mask

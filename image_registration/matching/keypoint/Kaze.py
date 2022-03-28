#! usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
from baseImage import Image, Rect
from baseImage.constant import Place

from image_registration.utils import generate_result
from image_registration.exceptions import NoEnoughPointsError, PerspectiveTransformError, HomographyError
from typing import Union, Tuple, List


class Kaze(object):
    FLANN_INDEX_KDTREE = 0
    FILTER_RATIO = 0.59
    METHOD_NAME = 'Kaze'
    Dtype = np.uint8
    Place = (Place.Mat, Place.UMat, Place.Ndarray)

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True):
        """
        初始化

        Args:
            threshold: 识别阈值(0~1)
            rgb: 是否使用rgb通道进行校验
        """
        self.threshold = threshold
        self.rgb = rgb
        self.detector = self.create_detector()
        self.matcher = self.create_matcher()

    def create_matcher(self) -> cv2.FlannBasedMatcher:
        """
        创建特征点匹配器

        Returns:
            cv2.FlannBasedMatcher
        """
        index_params = {'algorithm': self.FLANN_INDEX_KDTREE, 'tree': 5}
        # 指定递归遍历的次数. 值越高结果越准确，但是消耗的时间也越多
        search_params = {'checks': 50}
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        return matcher

    @staticmethod
    def create_detector() -> cv2.KAZE:
        """

        Returns:

        """
        detector = cv2.KAZE_create()
        return detector

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
        self.get_rect_from_good_matches(im_source=im_source, im_search=im_search, kp_src=kp_src, des_src=des_src,
                                        kp_sch=kp_sch, des_sch=des_sch)

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

        """
        matches = self.match_keypoint(des_sch=des_sch, des_src=des_src)
        good = self.get_good_in_matches(matches=matches)
        rect = self.extract_good_points(im_source=im_source, im_search=im_search, kp_sch=kp_sch, kp_src=kp_src, good=good)
        print(rect)
        # cv2.imshow('good', cv2.drawMatches(im_search.data, kp_sch, im_source.data, kp_src, good, None, flags=2))
        # cv2.waitKey(0)

    def match_keypoint(self, des_sch, des_src):
        """
        特征点匹配

        Args:
            des_src: 待匹配图像的描述符集
            des_sch: 图片模板的描述符集

        Returns:
            List[List[cv2.DMatch]]: 包含最匹配的描述符
        """
        # k=2表示每个特征点取出2个最匹配的对应点
        matches = self.matcher.knnMatch(des_sch, des_src, 2)
        return matches

    def get_good_in_matches(self, matches: list):
        """
        特征点过滤

        Args:
            matches: 特征点集

        Returns:
            List[cv2.DMatch]: 过滤后的描述符集
        """
        good = []
        for m, n in matches:
            if m.distance < self.FILTER_RATIO * n.distance:
                good.append(m)
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

    def input_image_check(self, im_source, im_search):
        im_source = self._image_check(im_source)
        im_search = self._image_check(im_search)

        assert im_source.place == im_search.place, '输入图片类型必须相同, source={}, search={}'.format(im_source.place, im_search.place)
        assert im_source.dtype == im_search.dtype, '输入图片数据类型必须相同, source={}, search={}'.format(im_source.dtype, im_search.dtype)
        assert im_source.channels == im_search.channels, '输入图片通道必须相同, source={}, search={}'.format(im_source.channels, im_search.channels)

        return im_source, im_search

    def _image_check(self, data: Union[str, bytes, np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat, Image]):
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

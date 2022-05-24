# import cv2
# import time
# import numpy as np
# import cupy as cp
# import ctypes
from baseImage import Image, Rect, Place, Setting
from image_registration.matching import CUDA_ORB, ORB, SIFT

from multiprocessing import Process, JoinableQueue
import time
import random


# im_source = Image('tests/image/keypoint_screen.png', place=Place.GpuMat).cvtColor(cv2.COLOR_BGR2GRAY)
# im_search = Image('tests/image/keypoint_search.png', place=Place.GpuMat).cvtColor(cv2.COLOR_BGR2GRAY)
#
# match = CUDA_ORB()
class AutoIncrement(object):
    def __init__(self):
        self._val = 0

    def __call__(self):
        self._val += 1
        return self._val


def match_process(queue: JoinableQueue, name: str):
    match = CUDA_ORB()
    while True:
        args = queue.get()
        im_source = Image(args['source'], place=Place.GpuMat)
        im_search = Image(args['search'], place=Place.GpuMat)
        res = match.find_best_result(im_source, im_search)
        print(f"res={res}, id={args['id']}, name={name}")
        queue.task_done()


index = AutoIncrement()


def producer(q, args: dict):
    for i in range(12):
        args = args.copy()
        args['id'] = index()
        q.put(args)

    q.join()


if __name__ == '__main__':
    q = JoinableQueue()

    p1 = Process(target=producer, args=(q, {"source": 'tests/image/keypoint_screen.png',
                                            "search": 'tests/image/keypoint_search.png'}))
    # 消费者们：即吃货们
    c1 = Process(target=match_process, args=(q, 'mike1'))
    c2 = Process(target=match_process, args=(q, 'mike2'))
    c3 = Process(target=match_process, args=(q, 'mike3'))
    c4 = Process(target=match_process, args=(q, 'mike4'))
    c1.daemon = True
    c2.daemon = True
    c3.daemon = True
    c4.daemon = True

    p1.start()
    c1.start()
    c2.start()
    c3.start()
    c4.start()

    p1.join()
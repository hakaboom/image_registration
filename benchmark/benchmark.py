from baseImage import Image
from baseImage.constant import Place
from image_registration.matching import SIFT, SURF, ORB, AKAZE, MatchTemplate, CudaMatchTemplate, CUDA_ORB
import time
import threading
import os
import json
import psutil


class RecordThread(threading.Thread):
    """记录CPU和内存数据的thread."""

    def __init__(self, interval=0.1):
        super(RecordThread, self).__init__()
        self.pid = os.getpid()
        self.interval = interval

        self.cpu_num = psutil.cpu_count()
        self.process = psutil.Process(self.pid)

        self.profile_data = []
        self.stop_flag = False

    def set_interval(self, interval):
        """设置数据采集间隔."""
        self.interval = interval

    def run(self):
        """开始线程."""
        while not self.stop_flag:
            timestamp = time.time()
            cpu_percent = self.process.cpu_percent() / self.cpu_num
            # mem_percent = mem = self.process.memory_percent()
            mem_info = dict(self.process.memory_info()._asdict())
            mem_gb_num = mem_info.get('rss', 0) / 1024 / 1024
            # 记录类变量
            self.profile_data.append({"mem_gb_num": mem_gb_num, "cpu_percent": cpu_percent, "timestamp": timestamp})
            # 记录cpu和mem_gb_num
            time.sleep(self.interval)


class Matching(object):

    METHODS = {
        'akaze': AKAZE(),
        'orb': ORB(),
        'sift': SIFT(),
        'surf': SURF(),
        'tpl': MatchTemplate(),
        'cuda_tpl': CudaMatchTemplate(),
        'cuda_orb': CUDA_ORB(),
    }

    def __init__(self, im_source, im_search):
        self.im_source = im_source
        self.im_search = im_search

    def _get_keypoint(self, method_name, im_source, im_search):
        func = self.METHODS[method_name]

        kp_src, des_src = func.get_keypoint_and_descriptor(image=im_source)
        kp_sch, des_sch = func.get_keypoint_and_descriptor(image=im_search)

        matches = func.match_keypoint(des_sch=des_sch, des_src=des_src)
        good = func.get_good_in_matches(matches=matches)
        # filtered_good_point, angle, first_point = func.filter_good_point(good=good, kp_src=kp_src, kp_sch=kp_sch)

        return kp_src, kp_sch, good

    def _get_tpl(self, method_name, im_source, im_search):
        func = self.METHODS[method_name]

        result = func.find_best_result(im_source, im_search)
        return result

    def run(self, method_name, im_source, im_search):
        if method_name in ['sift', 'surf', 'orb', 'akaze', 'cuda_orb']:
            return self._get_keypoint(method_name, im_source=im_source, im_search=im_search)
        elif method_name in ['tpl', 'cuda_tpl']:
            if im_source.place == Place.UMat:
                return None
            return self._get_tpl(method_name, im_source=im_source, im_search=im_search)


class Profile(object):
    def __init__(self, im_source, im_search, profile_interval=0.1):

        self.im_source = im_source
        self.im_search = im_search

        self.record_thread = RecordThread()
        self.record_thread.set_interval(profile_interval)
        self.matching_object = Matching(im_source, im_search)

        self.method_exec_info = []

    def run(self, method_list, counts=1):
        self.record_thread.stop_flag = False
        self.record_thread.start()

        for name in method_list:
            time.sleep(2)
            start_time = time.time()
            print("--->>> start '{}' matching:".format(name))
            ret_info = {
                'name': name,
            }
            if 'cuda' in name:
                source = Image(self.im_source, place=Place.GpuMat)
                search = Image(self.im_search, place=Place.GpuMat)
            elif 'opencl' in name:
                source = Image(self.im_source, place=Place.UMat)
                search = Image(self.im_search, place=Place.UMat)
                name = name.replace('opencl_', '')
            else:
                source = Image(self.im_source, place=Place.Ndarray)
                search = Image(self.im_search, place=Place.Ndarray)

            for i in range(counts):
                if 'tpl' in name:
                    result = self.matching_object.run(name, source, search)
                    kp_sch, kp_src, good = [], [], []
                else:
                    kp_sch, kp_src, good = self.matching_object.run(name, source, search)

            end_time = time.time()
            time.sleep(2)

            ret_info.update({
                'start_time': start_time,
                'end_time': end_time,
                'count': counts,
                'average': (end_time - start_time) / counts,
                'kp_sch': len(kp_sch),
                'kp_src': len(kp_src),
                'good': len(good)
            })
            print(ret_info)
            self.method_exec_info.append(ret_info)

        self.record_thread.stop_flag = True

    def write_to_json(self, dir_path="", file_name=""):
        data = {
            "plot_data": self.record_thread.profile_data,
            "method_exec_info": self.method_exec_info,
            "search_file": self.im_search,
            "source_file": self.im_source}
        # 写入文件
        file_path = os.path.join(dir_path, file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        json.dump(data, open(file_path, "w+"), indent=4)


if __name__ == '__main__':
    im_source = 'images/1.png'
    im_search = 'images/2.png'

    profile = Profile(im_source, im_search, 0.05)
    method_list = [
        'orb', 'cuda_orb', 'opencl_orb',
        'sift',
        'surf', 'opencl_surf',
        'akaze', 'opencl_akaze',
        'tpl', 'cuda_tpl'
    ]
    profile.run(method_list, counts=50)
    profile.write_to_json(dir_path="result", file_name="high_dpi.json")
# -*- coding : utf-8 -*-
# @Time     : 2023/5/16 - 15:09
# @Author   : enpei
# @FileName : utils.py
#
import time


class Times(object):
    def __init__(self):
        self.time = 0.
        # start time
        self.st = 0.
        # end time
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, repeats=1, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += (self.et - self.st) / repeats
        else:
            self.time = (self.et - self.st) / repeats

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


class Timer(Times):
    def __init__(self, with_tracker=False):
        super(Timer, self).__init__()
        self.with_tracker = with_tracker
        self.preprocess_time_s = Times()
        self.inference_time_s = Times()
        self.postprocess_time_s = Times()
        self.tracking_time_s = Times()
        self.img_num = 0

    def info(self, average=False):
        pre_time = self.preprocess_time_s.value()
        infer_time = self.inference_time_s.value()
        post_time = self.postprocess_time_s.value()
        track_time = self.tracking_time_s.value()

        total_time = pre_time + infer_time + post_time
        if self.with_tracker:
            total_time = total_time + track_time
        total_time = round(total_time, 4)
        print("------------------ Inference Time Info ----------------------")
        print("total_time(ms): {}, img_num: {}".format(total_time * 1000,
                                                       self.img_num))
        preprocess_time = round(pre_time / max(1, self.img_num),
                                4) if average else pre_time
        postprocess_time = round(post_time / max(1, self.img_num),
                                 4) if average else post_time
        inference_time = round(infer_time / max(1, self.img_num),
                               4) if average else infer_time
        tracking_time = round(track_time / max(1, self.img_num),
                              4) if average else track_time

        average_latency = total_time / max(1, self.img_num)
        qps = 0
        if total_time > 0:
            qps = 1 / average_latency
        print("average latency time(ms): {:.2f}, QPS: {:2f}".format(
            average_latency * 1000, qps))
        if self.with_tracker:
            print(
                "preprocess_time(ms): {:.2f}, inference_time(ms): {:.2f}, postprocess_time(ms): {:.2f}, tracking_time(ms): {:.2f}".
                format(preprocess_time * 1000, inference_time * 1000,
                       postprocess_time * 1000, tracking_time * 1000))
        else:
            print(
                "preprocess_time(ms): {:.2f}, inference_time(ms): {:.2f}, postprocess_time(ms): {:.2f}".
                format(preprocess_time * 1000, inference_time * 1000,
                       postprocess_time * 1000))

    def report(self, average=False):
        dic = {}
        pre_time = self.preprocess_time_s.value()
        infer_time = self.inference_time_s.value()
        post_time = self.postprocess_time_s.value()
        track_time = self.tracking_time_s.value()

        dic['preprocess_time_s'] = round(pre_time / max(1, self.img_num),
                                         4) if average else pre_time
        dic['inference_time_s'] = round(infer_time / max(1, self.img_num),
                                        4) if average else infer_time
        dic['postprocess_time_s'] = round(post_time / max(1, self.img_num),
                                          4) if average else post_time
        dic['img_num'] = self.img_num
        total_time = pre_time + infer_time + post_time
        if self.with_tracker:
            dic['tracking_time_s'] = round(track_time / max(1, self.img_num),
                                           4) if average else track_time
            total_time = total_time + track_time
        dic['total_time_s'] = round(total_time, 4)
        return dic

import os
from typing import Type, Union

import numpy as np
import torch

from .tracker import BaseTracker
from .deep_sort.deep_sort import DeepSort


class DEEPSORT(BaseTracker):
    def __init__(self, cfg: dict, *args, **kwargs):
        assert (
            "device" in kwargs
        ), "Used device type needs to be specified (cpu, gpu:0, gpu:1)!"

        # We need to jump through some hoops here to allow relative paths
        path = os.path.join(os.path.dirname(__file__), "..", cfg.DEEPSORT.REID_CKPT)
        path_resolved = os.path.abspath(path)
        self.DS = DeepSort(
            path_resolved,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            max_id=cfg.DEEPSORT.MAX_ID,
            use_cuda=True if kwargs["device"] != "CPU" else False,
        )

    def update(self, det: Type[torch.Tensor], img: Type[np.ndarray]) -> np.ndarray:
        """
        Tracker update function

        :param det (np.darray (,6)): detections array
        :param img (np.darray (,3)): original image

        :return (np.darray): [x1, y1, x2, y2, centre x, centre y, id]
        """
        bbox_xywh, confidences, _ = DEEPSORT.transform_detections(det)
        return self.DS.update(bbox_xywh, confidences, img)

    def get_tracker_flag(self):
        return self.DS.tracker.flag

    @staticmethod
    def transform_detections(det: Type[torch.Tensor]):
        """
        Transform detection vector to numpy

        :param det (torch.Tensor): prediction tensor

        :return xyxy (np.ndarray (,4)): x1, y1, x2, y2
                conf (np.ndarray (,1)): class confidences
                cls  (np,ndarray (,1)): class indeces
        """
        t_det = det.cpu().detach().numpy()
        return t_det[:, :4], t_det[..., 4], t_det[..., 5]

import numpy as np
import torch
from typing import Type

from .tracker import BaseTracker
from .deep_sort.deep_sort import DeepSort

class DEEPSORT(BaseTracker):
    def __init__(self, 
                 cfg: dict, 
                 *args, 
                 **kwargs):
        assert ('device' in kwargs
            ), "Used device type needs to be specified (cpu, gpu:0, gpu:1)!"

        self.DS = DeepSort(
            cfg.DEEPSORT.REID_CKPT, 
            max_dist=cfg.DEEPSORT.MAX_DIST, 
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
            max_age=cfg.DEEPSORT.MAX_AGE, 
            n_init=cfg.DEEPSORT.N_INIT, 
            nn_budget=cfg.DEEPSORT.NN_BUDGET, 
            use_cuda=True if kwargs['device'] != 'cpu' else False
        )

    def update(self, 
               det: Type[torch.Tensor], 
               img: Type[np.ndarray]
               ) -> np.ndarray:
        """
        Tracker update function

        :param det (np.darray): 
        :param img (np.darray): original image

        :return (np.darray):
        """
        t_det = DEEPSORT.transform_detections(det)

        bbox_xywh, confidences = t_det[:, :4], t_det[:, 4:]
        return self.DS.update(bbox_xywh, confidences, img)

    def get_tracker_flag(self):
        return self.DS.tracker.flag

    @staticmethod
    def transform_detections(det: Type[torch.Tensor]):
        t_det = det.cpu().detach().numpy()
        t_det = DEEPSORT.transform_xyxy_to_xywh(t_det)

        return t_det

    @staticmethod
    def transform_xyxy_to_xywh(det: Type[np.ndarray]):
        return det

